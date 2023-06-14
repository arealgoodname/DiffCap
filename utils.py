import torch.optim as optim
import torch
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

def get_optimizer(config, parameters):
    print("Using optimizer: {}".format(config.optim.optimizer))
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def get_lr_scheduler(optimizer, total_steps, config):
    def get_cosine_anneal_warmup_lambda(total_steps, step, top_anneal=10, repeat_times=5):
        assert total_steps % repeat_times == 0
        Ti = int(total_steps / repeat_times)
        Tc = step
        top_anneal_coeff = math.exp(-math.log(top_anneal) / (int(total_steps / repeat_times) * (repeat_times - 1)))
        cos_part = math.cos(math.pi * Tc / Ti)
        sin_part = math.sin(math.pi * Tc / Ti)
        if sin_part < 0:
            cos_part = -cos_part
        lr = 1 / 2 * (1 + cos_part) * (top_anneal_coeff ** step)
        return lr

    if config.optim.scheduler == 'CosineAnneal':
        repeat_times = config.optim.repeat_times
        assert total_steps % repeat_times == 0
        return CosineAnnealingLR(optimizer, T_max=int(total_steps / repeat_times))
    elif config.optim.scheduler == 'Linear':
        return LambdaLR(optimizer, lambda step: 1 - step / total_steps)
    elif config.optim.scheduler == 'Exp':
        decay_rate = config.optim.lr_decay_rate
        gamma = math.exp(-math.log(decay_rate)/total_steps)
        return LambdaLR(optimizer, lambda step: gamma ** step)
    elif config.optim.scheduler == 'CosineAnnealWarm':
        return LambdaLR(optimizer, lambda step: get_cosine_anneal_warmup_lambda(total_steps, step, config.optim.top_anneal, config.optim.repeat_times))
    else:
        raise NotImplementedError(
            'Scheduler {} not understood.'.format(config.optim.scheduler))

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def cond_fn(x, t_discrete, y, classifier, classifier_scale):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t_discrete)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


def generalized_steps(x, seq, model_fn, b, eta=0, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model_fn, b, is_cond_classifier=False, classifier=None, classifier_scale=1.0, **model_kwargs):
    with torch.no_grad():
        def model(x, t_discrete):
            if is_cond_classifier:
                y = model_kwargs.get("y", None)
                if y is None:
                    raise ValueError("For classifier guidance, the label y has to be in the input.")
                noise_uncond = model_fn(x, t_discrete, **model_kwargs)
                cond_grad = cond_fn(x, t_discrete, y, classifier=classifier, classifier_scale=classifier_scale)
                at = compute_alpha(b, t_discrete.long())
                sigma_t = (1 - at).sqrt()
                return noise_uncond - sigma_t * cond_grad
            else:
                return model_fn(x, t_discrete, **model_kwargs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def bleu(out, gt):
    smooooo = SmoothingFunction().method0
    assert type(out) == type(gt) == str
    out = [out]
    gt = [gt]
    return sentence_bleu([gt], out, smoothing_function=smooooo)

def meteor(out, gt):
    return single_meteor_score(set(out.split()), set(gt.split()))

def rougeL(out, gt):
    rg = Rouge()
    rouge_score = rg.get_scores(out, gt)[0]['rouge-l']['f']
    return rouge_score

