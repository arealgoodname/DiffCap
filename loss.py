import torch

def noise_estimation_loss(model, batch, t, e, b, keepdim=False, x0_predict=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1).to('cuda')
    #x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # the diffusion part goes into model since embedding is done there
    loss_mask = batch['loss_mask']
    loss_mask = loss_mask.unsqueeze(-1).repeat(1, 1, 768).to('cuda')
    if x0_predict:
        output, gt = model(noisy_input=None, t=t.float(), batch=batch, is_train=True, e=e, b=b, a=a)
        return ((output - gt) ** 2 * loss_mask).sum() / loss_mask.sum()
    else:
        output = model(noisy_input=None, t=t.float(), batch=batch, is_train=True, e=e, b=b, a=a)
        if keepdim:
            return ((e - output) * loss_mask).square().sum(dim=(1, 2))
        else:
            return ((e - output) * loss_mask).square().sum(dim=(1, 2)).mean(dim=0)

