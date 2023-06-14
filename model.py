import torch
import torch as th
import torch.nn as nn
from torch.nn import SiLU

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder as BertEncoder_for_Dlm

from bert import BertLayer
from collections import defaultdict
import copy

import math
import numpy as np


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros(size=(input_ids.shape[0], input_ids.shape[1]), dtype=torch.long).to(input_ids.device)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = (words_embeddings + position_embeddings + token_type_embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TextEmbeddingsforRoc(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = (words_embeddings + position_embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim=768, unconditional=False):
        super().__init__()
        self.img_transform = nn.Sequential(
            nn.Linear(img_dim, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        #self.img_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.unconditional = unconditional

    def forward(self, img_feat, pos_embs, type_embeddings):
        transformed_im = self.img_transform(img_feat)
        embeddings = transformed_im + type_embeddings + pos_embs
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddingsForDifussionLM(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, img_dim=768, unconditional=False):
        super().__init__()
        self.config = config
        self.embeddings = TextEmbeddings(config)
        self.image_embeddings = ImageEmbeddings(config, img_dim, unconditional)
        #self.unconditional = unconditional

    def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, pos_ids, img_type_ids=None):
        if img_type_ids is None:
            if len(img_feat.shape) == 2:
                img_feat_len = 1
            else:
                img_feat_len = img_feat.shape[1]
            img_type_ids = torch.ones(img_feat_len, dtype=torch.long, device=img_feat.device)
        img_position_embeddings = self.embeddings.position_embeddings(pos_ids)
        img_type_embeddings = self.embeddings.token_type_embeddings(img_type_ids)
        output = self.image_embeddings(img_feat, img_position_embeddings, img_type_embeddings)
        return output

    def _compute_img_txt_embeddings(self, img_feat, input_ids):
        if img_feat is None:
            img_feat_len = 0
        else:
            if len(img_feat.shape) == 2:
                img_feat_len = 1
            else:
                img_feat_len = img_feat.shape[1]
        position_ids = torch.arange(input_ids.shape[1] + img_feat_len, dtype=torch.long, device=input_ids.device)

        img_pos_ids = position_ids[:img_feat_len]
        txt_pos_ids = position_ids[img_feat_len:]

        txt_emb = self._compute_txt_embeddings(input_ids, txt_pos_ids)
        if img_feat is not None:
            img_emb = self._compute_img_embeddings(img_feat, img_pos_ids)
            if len(img_emb.shape) == 2:
                img_emb = img_emb.unsqueeze(1)
            embedding_output = torch.cat([img_emb, txt_emb], dim=1)
        else:
            embedding_output = txt_emb
        #head_cls_emb = self._compute_txt_embeddings(head_cls_t, head_cls_pos_ids)
        #timestep_emb = self.get_timestep_embedding(t, self.config.hidden_size)

        #embedding_output[:, 1, :] = time_emb # insert time embedding

        return embedding_output

    def forward(
            self,
            prefix,
            input_ids,
    ):
        '''
        attention_mask = torch.ones(size=(input_ids.shape[0], input_ids.shape[1] + img_feat.shape[1] + head_cls.shape[1]), device=input_ids.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        '''

        embedding_output = self._compute_img_txt_embeddings(prefix, input_ids)

        return embedding_output

class BertEmbeddingsForRoc(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TextEmbeddingsforRoc(config)

    def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids)
        return output

    def _compute_img_txt_embeddings(self, input_ids):
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids)
        embedding_output = txt_emb
        return embedding_output

    def forward(
            self,
            input_ids,
    ):

        attention_mask = torch.ones(size=(input_ids.shape[0], input_ids.shape[1]), device=input_ids.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self._compute_img_txt_embeddings(input_ids)

        return embedding_output, extended_attention_mask

class CapLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_type = config.model.feature_type
        self.embeddings = BertEmbeddingsForDifussionLM(config.bert,
                                                       img_dim=config.model.feature_dim,
                                                       unconditional=config.model.unconditional)
        bert_config = AutoConfig.from_pretrained('bert-base-uncased')
        bert_config.hidden_dropout_prob = config.bert.hidden_dropout_prob
        self.bert_encoder = BertEncoder_for_Dlm(bert_config)

        if config.training.encoder_reinit:
            print("===== reinitializing encoder =====")
            #for param in self.bert_encoder.parameters():
                #torch.nn.init.normal_(param) deprecated, this is not a good way to reinit
        else:
            print("===== loading Bert uncased weights =====")
            self.bert_encoder.load_state_dict(torch.load(config.bert.bert_encoder_path))

        self.word_embedding = torch.nn.Embedding(config.bert.vocab_size, config.bert.vocab_dim)
        #self.word_embedding.weight.data.normal_(mean=0.0)

        self.embeddings.embeddings.word_embeddings = nn.Sequential()
        self.config = config
        self.max_len = config.model.max_len
        self.lm_head = nn.Linear(config.bert.hidden_size, config.bert.vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        self.time_embedding = nn.Sequential(
            nn.Linear(config.bert.vocab_dim, min(config.bert.vocab_dim*4, config.bert.hidden_size)),
            SiLU(),
            nn.Linear(min(config.bert.vocab_dim*4, config.bert.hidden_size), config.bert.hidden_size),
        )

        if config.model.condition_method == 'add' or config.model.condition_method == 'prefix':
            self.condition_proj = nn.Sequential(
                nn.Linear(config.model.feature_dim, config.bert.hidden_size),
                nn.Tanh(),
                nn.Linear(config.bert.hidden_size, config.bert.hidden_size),)
        else:
            raise NotImplementedError
        self.input_up_proj = nn.Sequential(
            nn.Linear(config.bert.vocab_dim, config.bert.hidden_size),
            nn.Tanh(),
            nn.Linear(config.bert.hidden_size, config.bert.hidden_size),
        )

        self.output_down_proj = nn.Sequential(
            nn.Linear(config.bert.hidden_size, config.bert.hidden_size),
            nn.Tanh(),
            nn.Linear(config.bert.hidden_size, config.bert.vocab_dim),
        )

        if config.training.fp16:
            print("===== FP16 training=====")
        else:
            print("===== FP32 training=====")

    def get_sep_embedding(self, batch_size, sep_id):
        sep_embedding = self.txt_word_embeddings(torch.tensor([sep_id] * batch_size).to('cuda'))
        return sep_embedding

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_logits(self, sequence_output):
        logits = self.lm_head(sequence_output)
        return logits

    def forward(self, condition, t, x_t):
        x_embedding = self.input_up_proj(x_t)

        time_embedding = self.time_embedding(self.timestep_embedding(t, self.config.bert.vocab_dim))
        time_embedding = time_embedding.unsqueeze(1).expand(-1, x_embedding.shape[1], -1)

        txt_embs = x_embedding + time_embedding

        #cls_head = self.in_proj(self.get_sep_embedding(txt_embs.size(0), 0).unsqueeze(1))
        #sep_head = self.in_proj(self.get_sep_embedding(txt_embs.size(0), 1).unsqueeze(1))

        #sep_with_diffused = torch.cat([sep_head, txt_embs], dim=1)
        if self.config.model.condition_method == 'add':
            if len(condition.shape) == 2:
                condition = condition.unsqueeze(1).expand(-1, self.config.model.max_len, -1)
            else:
                raise NotImplementedError
            txt_embs = txt_embs + self.condition_proj(condition)
            embedding_output = self.embeddings(None, txt_embs)
        elif self.config.model.condition_method == 'prefix':
            embedding_output = self.embeddings(condition, txt_embs)
        else:
            raise NotImplementedError

        sequence_output = self.bert_encoder(embedding_output).last_hidden_state

        sequence_output = self.output_down_proj(sequence_output[:, -self.config.model.max_len:, :])

        return sequence_output

class SiLUbak(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class RocLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_type = config.model.feature_type
        self.word_embedding = nn.Embedding(config.bert.vocab_size, config.bert.vocab_dim)
        self.register_buffer("position_ids", torch.arange(config.bert.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.bert.max_position_embeddings, config.bert.hidden_size)


        bert_config = AutoConfig.from_pretrained('bert-base-uncased')
        bert_config.hidden_dropout_prob = config.bert.hidden_dropout_prob
        self.bert_encoder = BertEncoder_for_Dlm(bert_config)

        self.LayerNorm = nn.LayerNorm(config.bert.hidden_size, eps=bert_config.layer_norm_eps)
        self.dropout = nn.Dropout(config.bert.hidden_dropout_prob)

        print('======= ROC LM Online =======')
        if config.training.encoder_reinit:
            print("===== no weights loading for encoder =====")
        else:
            print("===== loading Bert uncased weights =====")
            self.bert_encoder.load_state_dict(torch.load(config.bert.bert_encoder_path))

        #self.txt_word_embeddings = torch.nn.Embedding(config.bert.vocab_size, config.bert.vocab_dim)
        #self.txt_word_embeddings.weight.data.normal_(mean=0.0)

        #self.embeddings.embeddings.word_embeddings = nn.Sequential()
        self.config = config
        self.max_len = config.model.max_len

        self.lm_head = nn.Linear(config.bert.vocab_dim, config.bert.vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        self.time_embedding = nn.Sequential(
            nn.Linear(config.bert.vocab_dim, config.bert.vocab_dim*4),
            SiLU(),
            nn.Linear(config.bert.vocab_dim*4, config.bert.hidden_size),
        )

        self.input_up_proj = nn.Sequential(
            nn.Linear(config.bert.vocab_dim, config.bert.hidden_size),
            nn.Tanh(),
            nn.Linear(config.bert.hidden_size, config.bert.hidden_size),
        )

        self.output_down_proj = nn.Sequential(
            nn.Linear(config.bert.hidden_size, config.bert.hidden_size),
            nn.Tanh(),
            nn.Linear(config.bert.hidden_size, config.bert.vocab_dim),
        )

        print('======= new timestep =======')

    def get_sep_embedding(self, batch_size, sep_id):
        sep_embedding = self.txt_word_embeddings(torch.tensor([sep_id] * batch_size).to('cuda'))
        return sep_embedding

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_logits(self, sequence_output):
        #mat = self.txt_word_embeddings.weight.unsqueeze(0).repeat(sequence_output.shape[0], 1, 1)
        #logits = torch.bmm(sequence_output, mat.transpose(1, 2))
        logits = self.lm_head(sequence_output)
        return logits

    def forward(self, condition, t, x_t):
        time_embedding = self.time_embedding(self.timestep_embedding(t, self.config.bert.vocab_dim))

        x_embedding = self.input_up_proj(x_t)
        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]

        txt_embs = self.position_embeddings(position_ids) + x_embedding + time_embedding.unsqueeze(1).expand(-1, self.config.model.max_len, -1)
        emb_inputs = self.dropout(self.LayerNorm(txt_embs))

        #embedding_output, extended_attention_mask = self.embeddings(txt_embs)

        sequence_output = self.bert_encoder(emb_inputs).last_hidden_state
        sequence_output = self.output_down_proj(sequence_output)

        return sequence_output

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
