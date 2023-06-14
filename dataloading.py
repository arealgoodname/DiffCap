import torch
from torchvision import transforms
from torch.utils.data import Dataset
import lmdb
import json
import msgpack
import msgpack_numpy
from lz4.frame import compress, decompress
from collections import defaultdict
from os.path import exists
import numpy as np
import horovod.torch as hvd
from PIL import Image
from torchvision.transforms import RandAugment, Resize, PILToTensor
msgpack_numpy.patch()

def _check_distributed():
    try:
        dist = hvd.size() != hvd.local_size()
    except ValueError:
        # not using horovod
        dist = False
    return dist

def get_ids_and_lens(db):
    assert isinstance(db, TxtTokLmdb)
    lens = []
    ids = []
    for id_ in list(db.id2len.keys())[hvd.rank()::hvd.size()]:
        lens.append(db.id2len[id_])
        ids.append(id_)
    return lens, ids

class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 readahead=not _check_distributed())
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret

class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=64, use_bert_tokenizer=False):
        self.fix_size = max_txt_len
        max_txt_len = max_txt_len - 1 # sep token
        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                           ).items()
                if len_ <= max_txt_len
            }
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        self.use_bert_tokenizer = use_bert_tokenizer
        if use_bert_tokenizer:
            self.cls_ = 101
            self.sep = 102
            self.mask = 103
        else:
            self.cls_ = 0
            self.sep = 1
            self.unknown = 2
            self.pad = 3

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump

    def combine_inputs(self, *inputs):
        input_ids = []
        if self.use_bert_tokenizer:
            for ids in inputs:
                input_ids.extend(ids + [self.sep] + [0] * (self.fix_size - len(ids) - 1))
        else:
            for ids in inputs:
                if len(ids) >= self.fix_size:
                    input_ids = ids[:self.fix_size]
                else:
                    input_ids = ids + [self.sep] + [self.pad] * (self.fix_size - len(ids) - 1)
                #if len(ids) > self.fix_size - 2:
                    #ids = ids[:self.fix_size - 2]
                #input_ids.extend(ids + [self.sep] + [self.pad] * (self.fix_size - len(ids) - 1))
        return torch.tensor(input_ids)

    @property
    def txt2img(self):
        txt2img = json.load(open(f'{self.db_dir}/txt2img.json'))
        return txt2img

    @property
    def img2txts(self):
        img2txts = json.load(open(f'{self.db_dir}/img2txts.json'))
        return img2txts

class ImageLmdbGroup(object):
    def __init__(self, npy_feature=True, unconditional=False):
        self.path2imgdb = {}
        self.npy_feature = npy_feature
        self.unconditional = unconditional

    def get_dataset_name(self, path):
        if 'flickr' in path:
            #print('flickr',path)
            return 'flickr'
        elif 'CC' in path:
            #print('cc', path)
            return 'cc'
        elif 'coco' in path:
            #print('coco', path)
            return 'coco'
        else:
            raise ValueError('unknown dataset')

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(img_dir=path,
                                    dataset=self.get_dataset_name(path),
                                    npy_feature=self.npy_feature,
                                    unconditional=self.unconditional)
        return img_db

class DetectFeatLmdb(object):
    def __init__(self, img_dir, dataset, npy_feature=False, unconditional=False):
        self.img_dir = img_dir
        self.resize = Resize((224, 224))
        self.randaug = RandAugment(2, 10)
        self.dataset = dataset
        self.npy_feature = npy_feature
        self.unconditional = unconditional

    def __getitem__(self, file_name):
        if self.unconditional:
            return torch.zeros(1)
        else:
            if self.dataset == 'coco':
                file_name = file_name.replace('coco', 'COCO')
                file_name = file_name.replace('npz', 'npy' if self.npy_feature else 'jpg')
            elif self.dataset == 'flickr':
                file_name = file_name.replace('npz', 'jpg').replace('flickr30k_', '')
                image_id, _ = file_name.split('.')
                file_name = str(int(image_id)) + ('.npy' if self.npy_feature else '.jpg')
            elif self.dataset == 'cc':
                pass
            else:
                raise ValueError('dataset not supported')
            if self.npy_feature:
                img = np.load(f'{self.img_dir}/{file_name}')
                img = torch.from_numpy(img)
            else:
                img = Image.open(f'{self.img_dir}/{file_name}').convert('RGB')
                img = self.resize(img)
                img = self.randaug(img)
            return img

class DetectFeatTxtTokDataset(Dataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        txt_lens, self.ids = get_ids_and_lens(txt_db)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def _get_img_feat(self, fname):
        img = self.img_db[fname]
        return img

class ItmRankDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, config):
        super().__init__(txt_db, img_db)
        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.one_to_one = config.data.one_to_one
        self.feature_type = config.model.feature_type
        if self.one_to_one:
            self.img2txts = {
                img: txts[0]
                for img, txts in self.img2txts.items()
            }
            self.ids = list(self.img2txts.values())
            self.txt2img = {id_: img for img, id_ in self.img2txts.items()}
        self.img_name_list = list(self.img2txts.keys())
        self.fix_len = config.model.fix_len
        if self.fix_len:
            self.max_len = config.model.max_len
        #self.t_head = config.model.t_head
        self.unconditional = config.model.unconditional
        self.use_bert_tokenizer = config.bert.use_bert_tokenizer
        if not self.use_bert_tokenizer:
            from spacy.lang.en import English
            import json
            nlp = English()
            self.tokenizer = nlp.tokenizer
            self.vocab = json.load(open(config.bert.vocab_pth))

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        if 'vit' in self.feature_type:
            gt_img_fname = self.txt2img[gt_txt_id]
        elif self.feature_type == 'clip_txt':
            gt_img_fname = str(gt_txt_id) + '.npy'
        else:
            gt_img_fname = None
        id_pairs = (gt_txt_id, gt_img_fname)
        inputs = self._collect_inputs(id_pairs)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        txt_id, img_id = id_pairs
        example = self.txt_db[txt_id]
        # text input
        if not self.use_bert_tokenizer:
            txt = example['raw']
            txt = self.tokenizer(txt)
            input_ids = [self.vocab.get(t.text, self.txt_db.unknown) for t in txt]
        else:
            input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        if not self.use_bert_tokenizer:
            loss_mask = torch.tensor([i != self.txt_db.unknown for i in input_ids])
        else:
            loss_mask = torch.ones(len(input_ids))
        # img input
        img_feat = self._get_img_feat(img_id)

        # input_ids = [ids, sep, pads]
        return input_ids, img_feat, loss_mask, img_id#torch.tensor(self.txt_db.sep), loss_mask