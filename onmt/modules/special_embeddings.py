import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from onmt.utils.logging import logger

class LookupEmbedding(nn.Module):
    def __init__(self, st, langs):
        """
        `langs` should be [ (ISO639P3, vocab index), ... ].
        """

        ### https://stackoverflow.com/a/43080779
        super(LookupEmbedding, self).__init__()

        self.st = st
        self.used_param_idxs = sorted(list(st.used_param_idxs))

        self.walsinfo = st.walsinfo
        self.langs = langs

        num_values = 0
        for param_idx in self.used_param_idxs:
            param_values_dict = self.walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx]

            param_name = self.walsinfo.param_idx_to_name[param_idx]
            param_id = self.walsinfo.param_idx_to_id[param_idx]
            param_space_size = len(param_values_dict)

            logger.info("Parameter {} (offset +{}):".format(param_idx, num_values))
            logger.info(" >> Name: {}".format(param_name))
            logger.info(" >> ID: {}".format(param_id))
            logger.info(" >> Parameter value space size: {}".format(param_space_size))

            for (param_value_idx, param_value_name) in param_values_dict.items():
                logger.info("     >> Parameter value {} (offset +{}), name: {}".format(param_value_idx, param_value_idx-1, param_value_name))
            
            num_values += param_space_size

        logger.info("Embedding dimension: {}".format(num_values))

        ### Maps vocab index of language to index in the embedding weight.
        ### First, calculate the lang ID token with the maximum vocab idx.
        max_lang_vocab_idx = -1
        for lang in langs:
            lang_vocab_idx = lang[1]
            if lang_vocab_idx > max_lang_vocab_idx:
                max_lang_vocab_idx = lang_vocab_idx
        logger.info("Maximum language ID token vocab idx: {}".format(max_lang_vocab_idx))

        lang_map = [ 0 ] * (max_lang_vocab_idx+1)

        for k, lang in enumerate(langs):
            logger.info("Lang ID {} ({}) maps to {}".format(lang[1], lang[0], k))
            lang_map[lang[1]] = k
        self.lang_map = torch.tensor(lang_map, dtype=torch.long).cuda()
        self.max_lang_vocab_idx = torch.tensor([max_lang_vocab_idx], dtype=torch.long).cuda()

        num_embeddings = len(langs)
        embedding_dim = num_values

        weight = torch.zeros((num_embeddings, embedding_dim)).cuda()
        self.gold = torch.zeros((num_embeddings, embedding_dim)).cuda()
        self.gold_mask = torch.zeros((num_embeddings, embedding_dim), dtype=torch.bool).cuda()

        for k, lang in enumerate(langs):
            iso639p3 = lang[0]
            try:
                lang_idx = st.iso639p3s.index(iso639p3)
            except ValueError:
                continue

            param_dict = st.param_dicts[lang_idx]

            offset = 0
            for param_idx in self.used_param_idxs:
                param_value_space_size = len(self.walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx])

                if param_idx in param_dict:
                    param_value_idx = param_dict[param_idx] - 1

                    weight[k][offset + param_value_idx] = 1.0
                    self.gold[k][offset + param_value_idx] = 1.0
                    self.gold_mask[k][offset : offset + param_value_space_size] = True

                offset += param_value_space_size

        self.embedding = nn.Embedding.from_pretrained(
            weight,
            freeze=False
        )

        def overwrite_gold(grad):
            return grad * self.gold_mask.logical_not()

        self.embedding.weight.register_hook(overwrite_gold)
        self.embedding.skip_init = True
        self.embedding.weight.skip_init = True

    def forward(self, x):
        """
        x_flat = x.flatten()
        y_flat = torch.empty_like(x_flat)

        #   Here we are converting the vocab idxs of language ID tokens into
        #   idxs into the lookup embeddings.
        for k, vocab_idx in enumerate(x_flat):
            #   The `.get(x.item(), 0)` part is basically saying that we should
            #   try and find the correct idx from `self.lang_map` if we can,
            #   and just use 0 if we can't find it. Why is this ok?
            #
            #   It's because the only cases when we can't find the correct idx
            #   for a vocab idx is when that vocab idx is not a language ID
            #   token. It's ok to take that to be the 0th (first) language ID
            #   token, since the embedding for this token will be masked out
            #   in `WordOrLanguageEmbedding.forward()` anyways.
            y_flat[k] = self.lang_map.get(vocab_idx.item(), 0)

        y = y_flat.reshape(x.shape)
        """

        #   Same work, but faster
        xcap = x.min(self.max_lang_vocab_idx)
        y = torch.index_select(self.lang_map, 0, xcap.flatten()).reshape(x.shape)

        return self.embedding(y)


class ParameterEmbedding(nn.Module):
    def __init__(self, st, emb_dim):
        super(ParameterEmbedding, self).__init__()

        self.st = st
        self.used_param_idxs = sorted(list(st.used_param_idxs))
        self.walsinfo = st.walsinfo

        num_values = 0
        for param_idx in self.used_param_idxs:
            num_values_per_param = len(self.walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx])
            num_values += num_values_per_param

        #   chanyoun says N(0, 0.001) is good:
        #   https://arxiv.org/pdf/1711.09160.pdf
        normal = Normal(0.0, 0.001)
        emb_weight = normal.sample((num_values, emb_dim))
        emb_weight.requires_grad = True
        # self.register_parameter(name="embedding", param=nn.Parameter(emb_weight.cuda()))
        self.embedding = nn.Parameter(emb_weight.cuda(), requires_grad=True)
        self.embedding.skip_init = True

        #   It's probably ok to just initialize weights to 1's
        num_params = len(self.used_param_idxs)
        #weights = torch.ones((num_params,), dtype=torch.float32, requires_grad=True)
        weights = torch.arange(num_params, dtype=torch.float32, requires_grad=True) / 10.0
        # self.register_parameter(name="weights", param=nn.Parameter(weights.cuda()))
        self.weights = nn.Parameter(weights.cuda(), requires_grad=True)
        self.weights.skip_init = True

        weight_idxs = []
        for k, param_idx in enumerate(self.used_param_idxs):
            num_values_per_param = len(self.walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx])
            weight_idxs += [ k for _ in range(num_values_per_param) ]
        self.weight_idxs = torch.tensor(weight_idxs, dtype=torch.long).cuda()

    #   Manual regularization of param weights.
    def maybe_regularize_weights(self):
        FACTOR = 10.0

        l1_norm = self.weights.sum().item()
        if l1_norm > self.weights.size(0) * FACTOR:
            self.weights.data /= l1_norm

    def forward(self, x):
        self.maybe_regularize_weights()

        #   self.weight:      [ w1, w2, w3 ]
        #   self.weight_idxs: [ 0,  0,  1,  1,  1,  2,  2,  2,  2  ]
        #   weight:           [ w1, w1, w2, w2, w2, w3, w3, w3, w3 ]
        weights = torch.index_select(self.weights, 0, self.weight_idxs)

        #   x.shape:  (d1, ..., dN, num_values)
        #   w.shape:  (num_values,)
        #   xw.shape: (d1, ..., dN, num_values)
        xw = x * weights

        #   emb.shape:  (num_values, emb_dim)
        #   combs.shape: (d1, ..., dN, emb_dim)
        embs = torch.tensordot(xw, self.embedding, dims=1)
        assert(embs.shape == x.shape[:-1] + self.embedding.shape[1:])

        return embs


class WordOrLanguageEmbedding(nn.Module):
    def __init__(self, word_embedding, lookup_embedding, param_embedding, lang_id_idxs):
        super(WordOrLanguageEmbedding, self).__init__()

        self.word_embedding = word_embedding
        self.lang_embedding = nn.Sequential(
            lookup_embedding,
            param_embedding
        )

        self.lang_id_idxs = torch.tensor(lang_id_idxs, dtype=torch.long).cuda()

    def forward(self, x):
        #   Which of indices in x are language ID tokens?
        # singleton_masks = [ x == lii for lii in self.lang_id_idxs ]
        # lang_mask = torch.stack(singleton_masks, dim=0).any(dim=0).unsqueeze(2).to(x.device)
        singleton_masks = x.unsqueeze(0) == self.lang_id_idxs.reshape((-1,) + (1,) * len(x.shape))
        lang_mask = singleton_masks.any(dim=0).unsqueeze(2).to(x.device)

        word_mask = lang_mask.logical_not()

        word_embeddings = self.word_embedding(x)
        lang_embeddings = self.lang_embedding(x)

        return word_embeddings * word_mask + lang_embeddings * lang_mask