import os
import math
import logging
import blobfile as bf
import json
import copy
import numpy as np
import tqdm
import torch

import horovod.torch as hvd

from model import CapLM, RocLM
from utils import get_optimizer
from loss import noise_estimation_loss
from dataloading import ItmRankDataset, ImageLmdbGroup, TxtTokLmdb
from torch.utils.data import DataLoader, ConcatDataset
from dataloader import build_dataloader, itm_rank_collate
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import get_lr_scheduler

import torchvision.utils as tvu

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class Diffusion(object):
    def __init__(self, args, config, device, model=None):
        self.args = args
        self.config = config
        self.device = device
        self.model_var_type = config.model.var_type
        self.model_mean_type = config.model.mean_type
        if config.diffusion.schedule_test:
            print('========== dpm schedule test ==========')
            betas = self.dpm_beta_schedule(config.diffusion.beta_schedule, config.diffusion.num_diffusion_timesteps)
        else:
            print('========== Lisa schedule ==========')
            betas = self.get_named_beta_schedule(config.diffusion.beta_schedule, config.diffusion.num_diffusion_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.model = model
        self.avg_eval_loss = None
        self.rescale_timesteps = config.diffusion.rescale_timesteps
        #self.total_steps = (93161//self.config.training.batch_size) * self.config.training.n_epochs

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def betas_for_alpha_bar2(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        betas.append(min(1 - alpha_bar(0), max_beta))
        for i in range(num_diffusion_timesteps - 1):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps):
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if schedule_name == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            print('############# linear #############')
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif schedule_name == "cosine":
            return self.betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        elif schedule_name == 'sqrt':
            print('############# sqrt #############')
            return self.betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: 1 - np.sqrt(t + 0.0001),
            )
        elif schedule_name == "trunc_cos":
            return self.betas_for_alpha_bar2(
                num_diffusion_timesteps,
                lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
            )
        elif schedule_name == 'trunc_lin':
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001 + 0.01
            beta_end = scale * 0.02 + 0.01
            return np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif schedule_name == 'pw_lin':
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001 + 0.01
            beta_mid = scale * 0.0001  # scale * 0.02
            beta_end = scale * 0.02
            first_part = np.linspace(
                beta_start, beta_mid, 10, dtype=np.float64
            )
            second_part = np.linspace(
                beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
            )
            return np.concatenate(
                [first_part, second_part]
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    def dpm_beta_schedule(self, beta_schedule, num_diffusion_timesteps, beta_start=None, beta_end=None):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_start:
            beta_start = beta_start
        else:
            beta_start = self.config.diffusion.beta_start
        if beta_end:
            beta_end = beta_end
        else:
            beta_end = self.config.diffusion.beta_end
        print('=========',beta_schedule,'=========')
        if beta_schedule == "quad":
            betas = (
                    np.linspace(
                        beta_start ** 0.5,
                        beta_end ** 0.5,
                        num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x0_helper(self, model_output, x, t):
        if self.model_mean_type == "start_x":
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
            return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}
        else:
            raise NotImplementedError

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def token_discrete_loss(self, x_t, get_logits, input_ids, loss_mask=None):
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids).to(x_t.device)
        logits = get_logits(x_t)  # bsz, seqlen, vocab
        # print(logits.shape)
        #assert torch.isnan(logits).sum() == 0
        #assert torch.isinf(logits).sum() == 0
        logits = logits.float()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = (loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape) * loss_mask).mean(dim=-1)
        # print(decoder_nll.shape)
        #lsm = torch.nn.LogSoftmax(dim=1)
        #nll = torch.nn.NLLLoss()
        #lsm_logits = lsm(logits)
        return decoder_nll

    def dlm_loss(self, model, x, t, cond):
        terms = {}
        input_ids = cond['input_ids'].to(x.device)
        x_start_mean = model.module.get_embeds(input_ids)

        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)

        x_start = x_start_mean + torch.randn_like(x_start_mean) * std
        x_t = self.q_sample(x_start, t)
        model_output = model(x_t, self._scale_timesteps(t))

        target = x_start
        temp_mse = self.mean_flat((target - model_output) ** 2)
        #assert torch.isnan(temp_mse).sum() == 0
        #assert torch.isinf(temp_mse).sum() == 0
        terms['mse'] = temp_mse

        if self.model_mean_type == 'start_x':
            model_out_x_start = model_output
        else:
            raise NotImplementedError

        t0_mask = (t == 0)
        t0_loss = self.mean_flat((x_start_mean - model_out_x_start) ** 2)

        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        #assert torch.isnan(out_mean).sum() == 0
        #assert torch.isinf(out_mean).sum() == 0
        tT_loss = self.mean_flat(out_mean ** 2)
        terms['tT'] = tT_loss

        #here i did some experiments
        decoder_nll = self.token_discrete_loss(x_start, model.module.get_logits, input_ids)
        #decoder_nll = self.token_discrete_loss(model_output, model.get_logits, input_ids)
        terms['nll'] = decoder_nll
        #assert torch.isnan(decoder_nll).sum() == 0
        #assert torch.isinf(decoder_nll).sum() == 0

        terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

        return terms

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def diffusion_lm_loss(self, model, batch, t):
        terms = {}
        input_ids = batch['input_ids'].to('cuda')
        loss_mask = batch['loss_mask'].to('cuda')
        condition = torch.stack(batch['imgs']).to('cuda')

        if 'vit' in self.config.model.feature_type:
            #assert condition.shape[1] == 197
            if len(condition.shape) == 3:
                assert condition.shape[1] == 197
            elif len(condition.shape) == 2:
                condition = condition.unsqueeze(1)
            if self.config.model.feature_type == 'vit_cls':
                condition = condition[:, 0, :]
        else:
            pass

        mse_loss_mask = loss_mask.unsqueeze(-1).expand(-1, -1, self.config.bert.vocab_dim)

        x_start_mean = model.word_embedding(input_ids)
        assert torch.isnan(x_start_mean).sum() == 0
        assert torch.isinf(x_start_mean).sum() == 0
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)

        x_start = x_start_mean + torch.randn_like(x_start_mean) * std
        x_t = self.q_sample(x_start, t)

        model_output = model(condition, self._scale_timesteps(t), x_t)

        #assert torch.isnan(model_output).sum() == 0
        target = x_start
        temp_mse = self.mean_flat(((target - model_output) ** 2) * mse_loss_mask)
        #assert torch.isnan(temp_mse).sum() == 0
        #assert torch.isinf(temp_mse).sum() == 0
        terms['mse'] = temp_mse

        if self.model_mean_type == 'start_x':
            model_out_x_start = model_output
        else:
            raise NotImplementedError

        t0_mask = (t == 0)
        t0_loss = self.mean_flat(((x_start_mean - model_out_x_start) ** 2) * mse_loss_mask)

        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        #assert torch.isnan(out_mean).sum() == 0
        #assert torch.isinf(out_mean).sum() == 0
        tT_loss = self.mean_flat((out_mean ** 2) * mse_loss_mask)
        terms['tT'] = tT_loss

        #here i did some experiments
        decoder_nll = self.token_discrete_loss(x_start, model.get_logits, input_ids, loss_mask)
        #decoder_nll = self.token_discrete_loss(model_output, model.get_logits, input_ids)
        terms['nll'] = decoder_nll
        #assert torch.isnan(decoder_nll).sum() == 0
        #assert torch.isinf(decoder_nll).sum() == 0

        terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

        return terms

    def get_roc_train_val_bak(self, config):
        import csv

        train_file = config.data.train_set
        val_file = config.data.test_set
        train_data = []
        val_data = []
        with open(train_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                train_data.append(line)

        with open(val_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                val_data.append(line)

        def collate_fn(inputs):
            inputs = list(map(lambda x: list(map(lambda y: int(y), x)), inputs))
            inputs = list(map(lambda x: torch.tensor(x).to(torch.int64), inputs))
            inputs = torch.stack(inputs)
            return_dict = {'input_ids': inputs}
            return return_dict

        train_loader = DataLoader(train_data, batch_size=config.training.batch_size,
                                  shuffle=True, num_workers=config.data.num_workers,
                                  pin_memory=config.data.pin_memory, collate_fn=collate_fn,
                                  drop_last=False)

        val_loader = DataLoader(val_data, batch_size=config.training.batch_size,
                                shuffle=True, num_workers=config.data.num_workers,
                                pin_memory=config.data.pin_memory, collate_fn=collate_fn,
                                drop_last=False)

        return train_loader, val_loader

    def get_roc_train_val(self, config):
        import argparse
        from improved_diffusion.text_datasets import load_data_text
        from improved_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser, str2bool
        def create_argparser():
            defaults = dict(
                data_dir="",
                schedule_sampler="uniform",
                lr=1e-4,
                weight_decay=0.0,
                lr_anneal_steps=0,
                batch_size=1,
                microbatch=-1,  # -1 disables microbatches
                ema_rate="0.9999",  # comma-separated list of EMA values
                log_interval=50,
                save_interval=50000,
                resume_checkpoint="",
                use_fp16=False,
                fp16_scale_growth=1e-3,
                seed=101,
                gradient_clipping=-1.0,
                eval_interval=2000,
                checkpoint_path='diff_models'
            )
            text_defaults = dict(modality='text',
                                 dataset_name='wikitext',
                                 dataset_config_name='wikitext-2-raw-v1',
                                 config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
                                 model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                                 experiment='gpt2_pre_compress', model_arch='conv-unet',
                                 roc_train='diffusion_lm/ROCstory',  # 'diffusion_lm/ROCstory/ROCstory17.csv',
                                 wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                                 e2e_train='e2e_data',
                                 yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                                 commonGen_train='diffusion_lm/common-gen/commongen_data',
                                 emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                                 padding_mode='block',
                                 preprocessing_num_workers=1)
            defaults.update(model_and_diffusion_defaults())
            defaults.update(text_defaults)
            parser = argparse.ArgumentParser()
            add_dict_to_argparser(parser, defaults)
            return parser
        args = create_argparser().parse_args([])
        args.batch_size = config.training.batch_size
        config_path = 'weights/training_args.json'
        with open(config_path, 'rb', ) as f:
            training_args = json.load(f)
        training_args['batch_size'] = config.training.batch_size

        args.__dict__.update(training_args)
        roc_data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            load_vocab=None,
            model=None,
        )

        data_valid = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split='valid',
            load_vocab=None,
            model=None,
        )
        return roc_data, data_valid

    def prepare_dataset(self, args, config):
        if config.data.dataset == 'roc':
            return self.get_roc_train_val(config)
        else:
            small_set = config.data.small_set
            if small_set:
                small_train_size, small_val_size = config.data.small_train_size, config.data.small_val_size
            all_img_dbs = ImageLmdbGroup(unconditional=config.model.unconditional)
            train_datasets = []
            for txt_path, img_path in zip(config.data.train_txt_dbs, config.data.train_img_dbs):
                img_db = all_img_dbs[img_path]
                txt_db = TxtTokLmdb(txt_path, config.model.max_len, use_bert_tokenizer=config.bert.use_bert_tokenizer)
                train_datasets.append(ItmRankDataset(txt_db, img_db, config))
            train_dataset = ConcatDataset(train_datasets)
            if small_set:
                train_dataset = torch.utils.data.Subset(train_dataset, range(small_train_size))
            train_sampler = DistributedSampler(train_dataset, num_replicas=len(args.gpus), rank=args.rank)
            train_dataloader = build_dataloader(train_dataset, itm_rank_collate, True, args, config, sampler=train_sampler)

            val_img_db = all_img_dbs[config.data.val_img_db]
            val_txt_db = TxtTokLmdb(config.data.val_txt_db, config.model.max_len, use_bert_tokenizer=config.bert.use_bert_tokenizer)
            val_dataset = ItmRankDataset(val_txt_db, val_img_db, config)
            if small_set:
                val_dataset = torch.utils.data.Subset(val_dataset, range(small_val_size))
            val_dataloader = build_dataloader(val_dataset, itm_rank_collate, True, args, config)
            if config.model.fix_len:
                print('============== fix len max len:', config.model.max_len,'==============')
            else:
                print('============== len varies ==============')
            return train_dataloader, val_dataloader

    def test_dataset(self, args, config):
        all_img_dbs = ImageLmdbGroup(unconditional=config.model.unconditional)
        val_img_db = all_img_dbs[config.data.test_img_db]
        val_txt_db = TxtTokLmdb(config.data.test_txt_db, config.model.max_len,
                                use_bert_tokenizer=config.bert.use_bert_tokenizer)
        val_dataset = ItmRankDataset(val_txt_db, val_img_db, config)
        val_dataloader = build_dataloader(val_dataset, itm_rank_collate, True, args, config)
        return val_dataloader

    def lr_anneal(self, step, opt):
        if step < self.config.training.warmup_steps:
            lr = step / self.config.training.warmup_steps
        else:
            frac_done = step / self.total_steps
            lr = self.config.optim.lr * (1 - frac_done)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def train_for_roc(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        train_data, test_data = self.prepare_dataset(args, config)
        # dataset, test_dataset = get_dataset(args, config)

        if self.model:
            model = self.model
        else:
            model = RocLM(config) if config.data.dataset == 'roc' else CapLM(config)
        model = model.to(self.device)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model_named_params = [{'params': p, 'layer_name': n} for n, p in model.named_parameters() if p.requires_grad]
        optimizer = get_optimizer(self.config, model_named_params)
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),)

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        #scaler = torch.cuda.amp.GradScaler(enabled=config.training.fp16)

        start_epoch, step = 0, 0
        if self.config.training.resume_training:
            print("============== Resuming training from checkpoint ================")
            states = torch.load(self.config.training.resume_pth)
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            report_on_start = True
        else:
            report_on_start = False
            print(" not resume training ")


        if hvd.rank() == 0:
            #pbar = tqdm(total=len(train_data) * self.config.training.n_epochs)
            pbar = tqdm(total=self.total_steps)
        for epoch in range(start_epoch, start_epoch + self.config.training.n_epochs):
            #for i, batch in enumerate(train_data):
            for _, batch in train_data:
                if step > self.total_steps:
                    break
                n = batch['input_ids'].size(0)
                model.train()
                step += 1

                optimizer.zero_grad()

                indices_np = np.random.choice(config.diffusion.num_diffusion_timesteps, size=(n,))
                t = torch.from_numpy(indices_np).long().to('cuda')

                losses = self.diffusion_lm_loss(model, batch, t)
                loss = losses["loss"].mean()

                tb_logger.add_scalar("loss", loss, global_step=step)
                tb_logger.add_scalar("mse", round(losses['mse'].mean().item(), 8), global_step=step)
                tb_logger.add_scalar("nll", round(losses['nll'].mean().item(), 8), global_step=step)
                tb_logger.add_scalar("tT", round(losses['tT'].mean().item(), 8), global_step=step)

                loss.backward()
                #scaler.scale(loss).backward()
                #optimizer.synchronize()

                #scaler.unscale_(optimizer)
                if config.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )

                if config.training.lr_anneal:
                    self.lr_anneal(step, optimizer)

                optimizer.step()

                if hvd.rank() == 0:
                    pbar.set_description(
                        f"loss: {round(loss.item(), 6)}, mse: {round(losses['mse'].mean().item(), 6)}, tT: {round(losses['tT'].mean().item(), 6)}, nll: {round(losses['nll'].mean().item(), 6)}")
                    pbar.update(1)

                '''
                if config.training.check_grad and config.training.fp16:
                    with torch.no_grad():
                        for group in optimizer.param_groups:
                            for param in group["params"]:
                                if param.grad is None:
                                    continue
                                if param.grad.is_sparse:
                                    if param.grad.dtype is torch.float16:
                                        param.grad = param.grad.coalesce()
                                    to_unscale = param.grad._values()
                                else:
                                    to_unscale = param.grad
                                v = to_unscale.clone().abs().max()
                                if torch.isinf(v):
                                    print('INF in', group['layer_name'], 'of step', step, '!!!')
                                if torch.isnan(v):
                                    print('NaN in', group['layer_name'], 'of step', step, '!!!')
                '''
                if step % self.config.training.snapshot_freq == 0 and step > 1 or report_on_start:
                    if hvd.rank() == 0:
                        #val_pbar = tqdm(total=len(test_data), desc="validating")
                        val_pbar = tqdm(total=5000//config.training.batch_size, desc="validating")
                    model.eval()
                    with torch.no_grad():
                        eval_losses = []
                        eval_step = 0
                        for _, batch in test_data:
                            if eval_step > 5000//config.training.batch_size:
                                break
                            n = batch['input_ids'].shape[0]
                            indices_np = np.random.choice(config.diffusion.num_diffusion_timesteps, size=(n,))
                            t = torch.from_numpy(indices_np).long().to('cuda')

                            loss = self.diffusion_lm_loss(model, batch, t)["loss"].mean()
                            eval_losses.append(loss.item())

                            tb_logger.add_scalar("val_loss", loss, global_step=step)
                            if hvd.rank() == 0:
                                val_pbar.update(1)
                            eval_step += 1

                    eval_loss = np.mean(eval_losses)
                    if self.avg_eval_loss is None:
                        self.avg_eval_loss = eval_loss
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_{}_{}.pth".format(step, eval_loss)))
                        logging.info("first eval loss {}".format(eval_loss))
                    elif eval_loss < self.avg_eval_loss:
                        self.avg_eval_loss = eval_loss
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_{}_{}.pth".format(step, eval_loss)))
                        if hvd.rank() == 0:
                            logging.info("improved eval loss {}".format(eval_loss))
                    else:
                        if hvd.rank() == 0:
                            logging.info("eval loss not improved: {} best is {}".format(eval_loss, self.avg_eval_loss))
                    report_on_start = False

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        train_data, test_data = self.prepare_dataset(args, config)
        # dataset, test_dataset = get_dataset(args, config)

        if self.model:
            model = self.model
        else:
            model = RocLM(config) if config.data.dataset == 'roc' else CapLM(config)
        model = model.to(self.device)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model_named_params = [{'params': p, 'layer_name': n} for n, p in model.named_parameters() if p.requires_grad]
        optimizer = get_optimizer(self.config, model_named_params)
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),)

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        #scaler = torch.cuda.amp.GradScaler(enabled=config.training.fp16)

        start_epoch, step = 0, 0
        if self.config.training.resume_training:
            print("============== Resuming training from checkpoint ================")
            states = torch.load(self.config.training.resume_pth)
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            report_on_start = True
        else:
            report_on_start = False
            print(" not resume training ")


        if hvd.rank() == 0:
            pbar = tqdm(total=len(train_data) * self.config.training.n_epochs)
            #pbar = tqdm(total=self.total_steps)
        self.total_steps = len(train_data) * self.config.training.n_epochs
        lr_scheduler =get_lr_scheduler(optimizer, total_steps=self.total_steps, config=config)

        prev_file = None
        for epoch in range(start_epoch, start_epoch + self.config.training.n_epochs):
            for i, batch in enumerate(train_data):
                n = batch['input_ids'].size(0)
                model.train()
                step += 1

                optimizer.zero_grad()

                indices_np = np.random.choice(config.diffusion.num_diffusion_timesteps, size=(n,))
                t = torch.from_numpy(indices_np).long().to('cuda')

                losses = self.diffusion_lm_loss(model, batch, t)
                loss = losses["loss"].mean()

                tb_logger.add_scalar("loss", loss, global_step=step)
                tb_logger.add_scalar("mse", round(losses['mse'].mean().item(), 8), global_step=step)
                tb_logger.add_scalar("nll", round(losses['nll'].mean().item(), 8), global_step=step)
                tb_logger.add_scalar("tT", round(losses['tT'].mean().item(), 8), global_step=step)

                loss.backward()
                #scaler.scale(loss).backward()
                #optimizer.synchronize()

                #scaler.unscale_(optimizer)
                if config.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )

                optimizer.step()
                lr_scheduler.step()
                tb_logger.add_scalar("lr", round(optimizer.param_groups[0]['lr'], 7), global_step=step)
                if hvd.rank() == 0:
                    pbar.set_description(
                        f"loss: {round(loss.item(), 4)}, mse: {round(losses['mse'].mean().item(), 4)}, "
                        f"tT: {round(losses['tT'].mean().item(), 4)}, nll: {round(losses['nll'].mean().item(), 4)}"
                        f"lr: {round(optimizer.param_groups[0]['lr'], 6)}")
                    pbar.update(1)

                '''
                if config.training.check_grad and config.training.fp16:
                    with torch.no_grad():
                        for group in optimizer.param_groups:
                            for param in group["params"]:
                                if param.grad is None:
                                    continue
                                if param.grad.is_sparse:
                                    if param.grad.dtype is torch.float16:
                                        param.grad = param.grad.coalesce()
                                    to_unscale = param.grad._values()
                                else:
                                    to_unscale = param.grad
                                v = to_unscale.clone().abs().max()
                                if torch.isinf(v):
                                    print('INF in', group['layer_name'], 'of step', step, '!!!')
                                if torch.isnan(v):
                                    print('NaN in', group['layer_name'], 'of step', step, '!!!')
                '''
                if step % self.config.training.snapshot_freq == 0 and step > 1 or report_on_start:
                    if hvd.rank() == 0:
                        val_pbar = tqdm(total=len(test_data), desc="validating")
                        #val_pbar = tqdm(total=5000//config.training.batch_size, desc="validating")
                    model.eval()
                    with torch.no_grad():
                        eval_losses = []
                        eval_step = 0
                        for _, batch in enumerate(test_data):
                            n = batch['input_ids'].shape[0]
                            indices_np = np.random.choice(config.diffusion.num_diffusion_timesteps, size=(n,))
                            t = torch.from_numpy(indices_np).long().to('cuda')

                            loss = self.diffusion_lm_loss(model, batch, t)["loss"].mean()
                            eval_losses.append(loss.item())

                            tb_logger.add_scalar("val_loss", loss, global_step=step)
                            if hvd.rank() == 0:
                                val_pbar.update(1)
                            eval_step += 1

                    eval_loss = np.mean(eval_losses)
                    if self.avg_eval_loss is None:
                        self.avg_eval_loss = eval_loss
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_{}_{}.pth".format(step, eval_loss)))
                        prev_file = os.path.join(self.args.log_path, "ckpt_{}_{}.pth".format(step, eval_loss))
                        logging.info("first eval loss {}".format(eval_loss))
                    elif eval_loss < self.avg_eval_loss:
                        self.avg_eval_loss = eval_loss
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_{}_{}.pth".format(step, eval_loss)))
                        if prev_file is not None:
                            os.remove(prev_file)
                        prev_file = os.path.join(self.args.log_path, "ckpt_{}_{}.pth".format(step, eval_loss))
                        if hvd.rank() == 0:
                            logging.info("improved eval loss {}".format(eval_loss))
                    else:
                        if hvd.rank() == 0:
                            logging.info("eval loss not improved: {} best is {}".format(eval_loss, self.avg_eval_loss))
                    report_on_start = False

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, condition, clip_denoised=True, denoised_fn=None):
        model_output = model(condition=condition, t=self._scale_timesteps(t), x_t=x)
        #model_output = model(x, t)
        model_variance, model_log_variance = (
            self.posterior_variance,
            self.posterior_log_variance_clipped
        )

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                #x = denoised_fn(model.word_embedding, x, t)
                x = denoised_fn(model.word_embedding, x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(model_output)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, t, condition, clip_denoised=False, denoised_fn=None, langevin_fn=None):
        out = self.p_mean_variance(
            model,
            x,
            t,
            condition,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean': out["mean"], 'out': out}

    def denoised_fn_round(self, model, text_emb, t):
        down_proj_emb = model.weight  # input_embs
        # print(t)
        old_shape = text_emb.shape
        old_device = text_emb.device

        def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
            if dist == 'l2':
                emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # vocab
                text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
                arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
                # print(emb_norm.shape, arr_norm.shape)
                dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb,
                                                                         text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
                dist = torch.clamp(dist, 0.0, np.inf)
                # print(dist.shape)
            topk_out = torch.topk(-dist, k=1, dim=0)
            return topk_out.values, topk_out.indices

        dist = 'l2'
        if len(text_emb.shape) > 2:
            text_emb = text_emb.reshape(-1, text_emb.size(-1))
        else:
            text_emb = text_emb
        # val, indices = get_knn(down_proj_emb,
        #                        text_emb.to(down_proj_emb.device), dist=dist)
        val, indices = get_efficient_knn(down_proj_emb,
                                         text_emb.to(down_proj_emb.device), dist=dist)
        rounded_tokens = indices[0]
        # print(rounded_tokens.shape)
        new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
        return new_embeds

    def sample_decode(self, sample, model, rev_vocab, gt_ids=None):
        with torch.no_grad():
            true_logits = model.get_logits(sample)
            cands = torch.topk(true_logits, k=1, dim=-1).indices.squeeze(-1)

        ids = cands.detach().cpu().numpy()

        if gt_ids is None:
            gt_ids = [[] for _ in range(len(ids))]
        else:
            gt_ids = gt_ids.detach().cpu().numpy()
        decoded = []
        gt_decoded = []
        for line, gt_id in zip(ids, gt_ids):
            line = [rev_vocab[i] for i in line]
            gt_id = [rev_vocab[i] for i in gt_id]
            decoded.append(line)
            gt_decoded.append(gt_id)
        return decoded, gt_decoded

    def sample(self, args, config):
        print('new sampling with rescale')
        import json
        with open(config.bert.vocab_pth, 'r') as f:
            vocab = json.load(f)
        rev_vocab = {v: k for k, v in vocab.items()}
        if 1 in rev_vocab.keys():
            pass
        else:
            rev_vocab[1] = '[END]'

        if self.model:
            print('============== got external model ===============')
            model = self.model
        else:
            model = RocLM(config) if config.data.dataset == 'roc' else CapLM(config)
            model.load_state_dict(torch.load(config.model.load_path))
        if config.model.unconditional:
            print('============== unconditional sampling ===============')
            test_set = range(1)
        else:
            test_set = self.test_dataset(args, config)
        decodeds = []
        gts = []
        gt_ids = []
        for batch, _ in zip(test_set, range(config.sampling.total_samples)):
            if config.model.unconditional:
                condition = None
                gt = None
                gt_id = None
                sample_shape = (config.sampling.batch_size, config.model.max_len, config.bert.vocab_dim)
            else:
                condition = torch.stack(batch['imgs']).to('cuda')
                if 'vit' in self.config.model.feature_type:
                    # assert condition.shape[1] == 197
                    if len(condition.shape) == 3:
                        assert condition.shape[1] == 197
                    elif len(condition.shape) == 2:
                        condition = condition.unsqueeze(1)
                    if self.config.model.feature_type == 'vit_cls':
                        condition = condition[:, 0, :]
                else:
                    pass
                gt = batch['input_ids']
                gt_id = batch['img_id']
                sample_shape = (condition.shape[0], config.model.max_len, config.bert.vocab_dim)
            batch_sample = self.sample_batch(model, sample_shape, config.diffusion.num_diffusion_timesteps, condition)
            decoded_batch, gt_batch = self.sample_decode(batch_sample, model, rev_vocab, gt)
            decodeds.append(decoded_batch)
            gts.append(gt_batch)
            gt_ids.append(gt_id)
        return decodeds, gts, gt_ids

    def sample_mid(self, args, config):
        print('sampling with middle output')
        import json
        with open(config.bert.vocab_pth, 'r') as f:
            vocab = json.load(f)
        rev_vocab = {v: k for k, v in vocab.items()}
        if 1 in rev_vocab.keys():
            pass
        else:
            rev_vocab[1] = '[END]'

        if self.model:
            print('============== got external model ===============')
            model = self.model
        else:
            model = RocLM(config) if config.data.dataset == 'roc' else CapLM(config)
            model.load_state_dict(torch.load(config.model.load_path))
        if config.model.unconditional:
            print('============== unconditional sampling ===============')
            test_set = range(1)
        else:
            test_set = self.test_dataset(args, config)
        decodeds = []
        gts = []
        gt_ids = []
        for batch, _ in zip(test_set, range(config.sampling.total_samples)):
            if config.model.unconditional:
                condition = None
                gt = None
                gt_id = None
                sample_shape = (config.sampling.batch_size, config.model.max_len, config.bert.vocab_dim)
            else:
                condition = torch.stack(batch['imgs']).to('cuda')

                gt = batch['input_ids']
                gt_id = batch['img_id']
                sample_shape = (condition.shape[0], config.model.max_len, config.bert.vocab_dim)
            batch_sample = self.sample_batch(model, sample_shape, config.diffusion.num_diffusion_timesteps, condition, with_middle=True)
            mid_decodeds = []
            mid_gts = []
            mid_ids = []
            for mid_sample in batch_sample:
                decoded_batch, gt_batch = self.sample_decode(mid_sample, model, rev_vocab, gt)
                mid_decodeds.append(decoded_batch)
                mid_gts.append(gt_batch)
                mid_ids.append(gt_id)
            decodeds.append(mid_decodeds)
            gts.append(mid_gts)
            gt_ids.append(mid_ids)
        return decodeds, gts, gt_ids

    def post_process(self, word_lst):
        start = ''
        for word in word_lst:
            if word != '[END]':
                start += (word + ' ')
            else:
                break
        return start

    def sample_batch(self, model, sample_shape, steps, condition=None, sample_fn=None, langevin_fn=None, with_middle=False):
        rand_noise = torch.randn(*sample_shape).to(next(model.parameters()).device)
        indices = list(range(steps))[::-1]
        if with_middle:
            outs = []
        if sample_fn is None:
            sample_fn = self.p_sample
        for i in tqdm(indices):
            t = torch.tensor([i] * sample_shape[0]).to(next(model.parameters()).device)
            with torch.no_grad():
                out = sample_fn(
                    model,
                    rand_noise,
                    t,
                    condition,
                    denoised_fn=self.denoised_fn_round,
                    langevin_fn=langevin_fn,
                )
                rand_noise = out["sample"]
                if with_middle:
                    outs.append(rand_noise)
        samples = out["sample"]
        if with_middle:
            return outs
        return samples

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def langevin_fn(
            self,
            clip_processor,
            clip_model,
            pils,
            rev_vocab,
            model,
            sample,
            sigma,
            mean,
            K=3,
            lr=0.1,
    ):
        input_emb_param = torch.nn.Parameter(sample)
        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_emb_param], lr=lr)
                optimizer.zero_grad()
                decoded_batch, _ = self.sample_decode(sample, model, rev_vocab)
                decoded_batch = [self.post_process(line) for line in decoded_batch]

                energy_input = clip_processor(text=decoded_batch, images=pils, return_tensors='pt', padding=True).to(clip_model.device)
                energy = clip_model(**energy_input, return_loss=True)

                coef = 0.0001
                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_emb_param) ** 2 / 1.).mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_emb_param) ** 2 / sigma).mean(dim=0).sum()
                loss = energy.loss + logp_term

                loss.backward()
                optimizer.step()

                epsilon = torch.randn_like(input_emb_param.data)
                input_emb_param = torch.nn.Parameter(
                    (input_emb_param.data + 0.0 * sigma.mean().item() * epsilon).detach())
        return input_emb_param.data


    def ddim_sample(
            self,
            model,
            x,
            t,
            condition=None,
            clip_denoised=True,
            denoised_fn=None,
            langevin_fn=None,
            eta=0.0,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            condition=condition
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn is not None:
            sample = langevin_fn(model, sample, sigma, mean_pred)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}