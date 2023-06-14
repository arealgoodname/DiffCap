"""
code from uniter model
thanks to their contribution
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
from torch.utils.data import DataLoader

def build_dataloader(dataset, collate_fn, is_train, args, config, sampler=None):
    dataloader = DataLoader(dataset, batch_size=(config.training.batch_size if is_train else config.sampling.batch_size),
                            shuffle=(False if sampler else True), drop_last=False,
                            num_workers=config.data.num_workers,
                            pin_memory=config.data.pin_memory, collate_fn=collate_fn, sampler=sampler)
    dataloader = PrefetchLoader(dataloader)
    return dataloader

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch

def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass

class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def itm_rank_collate(inputs):
    input_ids, imgs, loss_mask, img_id = list(map(list, unzip(concat(i for i in [inputs]))))
    #print('got loss: mask', loss_mask)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    #sep = torch.stack(sep, dim=0).unsqueeze(-1)
    #input_ids = torch.cat([input_ids, sep], dim=1)
    #position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    loss_mask = pad_sequence(loss_mask, batch_first=True, padding_value=True)

    _, max_txt_len = input_ids.size()


    batch = {
        'imgs': imgs,
        'input_ids': input_ids,
        #'cls_t_head': cls_t_head,
        #'position_ids': position_ids,
        #'txt_len': max_txt_len - 1,
        'loss_mask': loss_mask,
        'img_id': img_id,
             }
    return batch
