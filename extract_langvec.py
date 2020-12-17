import os
import argparse
import torch
import pickle
import json
from pprint import pprint
from tqdm import tqdm
import onmt.inputters as inputters
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_base_model
from onmt.modules.special_embeddings import LookupEmbedding, ParameterEmbedding, WordOrLanguageEmbedding

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", required=True, type=str,
                    help="path to the trained model")
parser.add_argument("-out", "--out", required=True, type=str,
                    help="dir path to save the resulting pickle file")
parser.add_argument("-data", "--data", required=True, type=str,
                    help="path to dir that contains LANG_en text files")
parser.add_argument("-mode", "--mode", type=str, default="eng-as-tgt-lang",
                    choices=["eng-as-src-lang", "eng-as-tgt-lang", "eng-as-src-nolang", "eng-as-tgt-nolang"], help="translate mode")
parser.add_argument("-wals", "--wals", type=str, default="/home/ubuntu/project11737-reprise/sozaki/realdata/wals",
                    help="path to wals data")
parser.add_argument("-sigtyp", "--sigtyp", type=str, default="/home/ubuntu/project11737-reprise/sozaki/realdata/sigtyp",
                    help="path to sigtyp dataset")
parser.add_argument("-save_lang_embed_info",
                    "--save_lang_embed_info", type=str)
parser.add_argument("-gpu_id", "--gpu_id", type=int, default=0,
                    help="which gpu to use, set to -1 to use CPU")
args = parser.parse_args()

assert os.path.exists(args.model), f"model {args.model} does not exist"
assert os.path.isdir(
    args.out), f"model {args.out} should be an existing directory"
assert os.path.exists(os.path.join(args.data, "model.vocab.special")
                      ), f"vocab.special model does not exist under your bpe data folder ({args.data})"
device = torch.device("cuda", args.gpu_id) if (
    args.gpu_id > -1) else torch.device("cpu")

if os.path.isdir(args.model) and not os.path.isfile(args.model):
    with open(os.path.join(args.model, "modelbest_ckpt_config.json"), "r") as file:
        best_model = json.load(file)
    best_modelname = f"model_step_{best_model['step']}_{best_model['validation_acc']:.2f}"
    name_best_checkpoint = f"{best_modelname}.pt"
    args.model = os.path.join(args.model, name_best_checkpoint)
else:
    best_modelname = args.model.replace(".pt", "").split("/")[-1]

if not os.path.isdir(os.path.join(args.out, best_modelname)):
    os.mkdir(os.path.join(args.out, best_modelname))
args.out = os.path.join(args.out, best_modelname)

# functions


def load_model(args):
    print(f"loading best model from {args.model}")
    checkpoint = torch.load(args.model,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    fields = checkpoint['vocab']
    model = build_base_model(model_opt, fields, (args.gpu_id != -1), checkpoint,
                             args.gpu_id)
    model.eval()
    model.generator.eval()
    return checkpoint, fields, model, model_opt


def get_langvec_from_enc_states(enc_states, use_hidden=False, contextualized=True):
    # TODO: automaticallly extract these from model_opt
    NUM_LAYERS, NUM_DIRECTIONS, HIDDEN_SIZE = 2, 2, 256
    hidden, cell = enc_states
    direction_index = 1 if contextualized else 0
    if use_hidden:
        _hidden = hidden.view(
            NUM_LAYERS, NUM_DIRECTIONS, -1, HIDDEN_SIZE).cpu()
        _hidden_direction = _hidden[:, direction_index]
        return _hidden_direction[0].sum(dim=0), _hidden_direction[1].sum(dim=0)
    else:
        _cell = cell.view(NUM_LAYERS, NUM_DIRECTIONS, -1, HIDDEN_SIZE).cpu()
        _cell_direction = _cell[:, direction_index]

        return _cell_direction[0].sum(dim=0), _cell_direction[1].sum(dim=0)


def run_model_for_one_lang(args, model, fields, path_data, batch_size=50, shard_size=0):
    src_shards = split_corpus(path_data, shard_size)
    total_vec = {"cell": {"context1": None, "context2": None,
                          "uncontext1": None, "uncontext2": None},
                 "hidden": {"context1": None, "context2": None,
                            "uncontext1": None, "uncontext2": None}}
    total_num_vec = 0
    for src_shard in src_shards:
        src_reader = inputters.str2reader["text"].from_opt(args)
        _readers, _data = inputters.Dataset.config(
            [("src", {"reader": src_reader, "data": src_shard})]
        )

        data = inputters.Dataset(fields, readers=_readers, data=_data,
                                 sort_key=inputters.str2sortkey["text"], filter_pred=None,)

        data_iter = inputters.OrderedIterator(dataset=data, device=device, batch_size=batch_size,
                                              batch_size_fn=None, train=False, sort=False, sort_within_batch=True, shuffle=False,)

        batch_num = 0
        with torch.no_grad():
            for batch in tqdm(data_iter, total=2000):
                if batch_num > 2000:
                    break
                batch_num += 1
                src, src_lengths = (
                    batch.src if isinstance(
                        batch.src, tuple) else (batch.src, None)
                )

                enc_states = {}
                enc_states["context"], _, _ = model.encoder(
                    src, src_lengths
                )
                src = src[0].unsqueeze(0)
                src_lengths = torch.ones_like(src_lengths)

                enc_states["uncontext"], _, _ = model.encoder(
                    src, src_lengths
                )

                for mode in ["cell", "hidden"]:
                    for context_mode in ["context", "uncontext"]:
                        vec1, vec2 = get_langvec_from_enc_states(enc_states[context_mode], use_hidden=(
                            mode == "hidden"), contextualized=(context_mode == "context"))
                        for layer_mode, layer_vec in zip(["1", "2"], [vec1, vec2]):
                            if total_vec[mode][context_mode+layer_mode] is None:
                                total_vec[mode][context_mode +
                                                layer_mode] = layer_vec
                            else:
                                total_vec[mode][context_mode +
                                                layer_mode] = total_vec[mode][context_mode+layer_mode].add(layer_vec)
                total_num_vec += len(batch)

    for mode in ["cell", "hidden"]:
        for context_mode in ["context", "uncontext"]:
            for layer_mode in ["1", "2"]:
                total_vec[mode][context_mode+layer_mode] /= total_num_vec
    return total_vec, total_num_vec


# loading model & data
model, fields, model_run, model_opt = load_model(args)
print("================================================================================")
print("Parameters in the model:")
pprint(list(model['model'].keys()))
print("")

with open(os.path.join(args.data, "model.vocab.special"), "r") as file:
    langs = [l.strip().replace("LANG_", "").lower()
             for l in file.readlines() if l != "\n"]
print("langs: ", langs)


# extract trained embedding & make predictions for sigtyp test set
common_part = ".embeddings.make_embedding.emb_luts."
word_emb_weight_part = "0.word_embedding.weight"
lookup_emb_weight_part = "0.lang_embedding.0.embedding.weight"
param_emb_part = "0.lang_embedding.1.embedding"
param_weights_part = "0.lang_embedding.1.weights"

print("================================================================================")
print("Loading weights...")
print("")
enc_word_emb_weight = model["model"]["encoder" +
                                     common_part + word_emb_weight_part]
enc_lookup_emb_weight = model["model"]["encoder" +
                                       common_part + lookup_emb_weight_part]
enc_param_emb = model["model"]["encoder" + common_part + param_emb_part]
enc_param_weights = model["model"]["encoder" +
                                   common_part + param_weights_part]

dec_word_emb_weight = model["model"]["decoder" +
                                     common_part + word_emb_weight_part]
dec_lookup_emb_weight = model["model"]["decoder" +
                                       common_part + lookup_emb_weight_part]
dec_param_emb = model["model"]["decoder" + common_part + param_emb_part]
dec_param_weights = model["model"]["decoder" +
                                   common_part + param_weights_part]


def diff(t1, t2):
    return (t1 - t2).abs().sum().item()


print("================================================================================")
print("Differences between encoder and decoder weights (low diff means weight sharing worked):")
print("word_emb_weight:",   diff(enc_word_emb_weight,   dec_word_emb_weight))
print("lookup_emb_weight:", diff(enc_lookup_emb_weight, dec_lookup_emb_weight))
print("param_emb:",         diff(enc_param_emb,         dec_param_emb))
print("param_weights:",     diff(enc_param_weights,     dec_param_weights))
print("")

# run the model with LANG_en train data, and get avg. cell state
print("================================================================================")
print("Loading map info...")
print("")

info = torch.load(args.save_lang_embed_info)
langs = info["langs"]
sigtyp = info["sigtyp"]
walsinfo = info["walsinfo"]

emb_dim = enc_param_emb.size(1)

lookup_embedding = LookupEmbedding(sigtyp, langs)
param_embedding = ParameterEmbedding(sigtyp, emb_dim)

# swap out the tensors
lookup_embedding.embedding.weight.data = enc_lookup_emb_weight.to(device)
param_embedding.embedding.data = enc_param_emb.to(device)
param_embedding.weights.data = enc_param_weights.to(device)

# language embeddings
lang_embs = dict()

print("================================================================================")
print("Getting language embeddings ({}d)...".format(emb_dim))
print("")

vocab_idxs = []
for (iso639p3, vocab_idx) in langs:
    vocab_idxs.append(vocab_idx)

vocab_idxs_batch = torch.tensor(vocab_idxs, dtype=torch.long).to(device)
lookup_batch = lookup_embedding(vocab_idxs_batch)
lang_embs_batch = param_embedding(lookup_batch)

for (iso639p3, _), lang_emb in zip(langs, lang_embs_batch):
    lang_embs[iso639p3] = lang_emb

pprint(lang_embs)

print("================================================================================")
print("Updating typological predictions...")
print("")

used_param_idxs = sorted(list(sigtyp.used_param_idxs))

for (iso639p3, _), lookup in zip(langs, lookup_batch):
    print("Processing language {}...".format(iso639p3))

    try:
        sigtyp_idx = sigtyp.iso639p3s.index(iso639p3)
    except ValueError:
        sigtyp_idx = len(sigtyp.iso639p3s)

        lang_name = "Language {}".format(iso639p3)
        print("Language {} not found in Sigtyp. Adding it to the end, with name=\"{}\", genus=0, family=0".format(
            iso639p3, lang_name))

        sigtyp.iso639p3s.append(iso639p3)
        sigtyp.names.append(lang_name)
        sigtyp.genus_idxs.append(0)
        sigtyp.family_idxs.append(0)
        sigtyp.param_dicts.append(dict())

    offset = 0
    for param_idx in used_param_idxs:
        param_values_dict = walsinfo.param_idx_to_param_value_idx_to_param_value_name[
            param_idx]
        size = len(param_values_dict)

        pred = lookup[offset: offset + size].argmax().item() + 1
        pred_prob = lookup[offset: offset + size][pred-1].item()
        if param_idx in sigtyp.param_dicts[sigtyp_idx]:
            sigtyp_gold = sigtyp.param_dicts[sigtyp_idx][param_idx]

            if pred != sigtyp_gold:
                print("WRONG: Sigtyp gold is {}, pred is {}".format(
                    sigtyp_gold, pred))
                # raise Exception()
        else:
            param_name = walsinfo.param_idx_to_name[param_idx]
            print("[{}] No gold value for parameter {} \"{}\",".format(
                iso639p3, param_idx, param_name))
            print("    filling in with prediction {} \"{}\"...".format(
                pred, param_values_dict[pred]))
            print("    Original lookup: {}".format(
                lookup[offset: offset + size].cpu().tolist()))
            sigtyp.param_dicts[sigtyp_idx][param_idx] = pred
            if sigtyp_idx not in sigtyp.pred_param_dicts:
                sigtyp.pred_param_dicts[sigtyp_idx] = {}
            sigtyp.pred_param_dicts[sigtyp_idx][param_idx] = (
                pred, round(pred_prob, 3))

        offset += size

# run the model with LANG_en train data, and get avg. cell state
# lang_vec_cell_uncontext, lang_vec_cell_context = {}, {}
lang_rnn_vec = {}
for lang, lang_idx in langs:
    path_lang = os.path.join(args.data, lang+"-eng", args.mode)
    path_train_src, path_train_tgt = os.path.join(
        path_lang, "src-train.txt"), os.path.join(path_lang, "tgt-train.txt")
    rnn_vectors, total_num_vec = run_model_for_one_lang(
        args, model_run, fields, path_train_src)
    lang_rnn_vec[lang] = rnn_vectors


# save resulting langvecs (lang_vec_emb, lang_vec_cell_uncontext, lang_vec_cell_context)
# output should be a langvec dict something like {"deu":[0.1, 0.03, ...], "kor": [-0.2, 0.01, ...]}
with open(os.path.join(args.out, "typvec.pkl"), "wb") as file:
    pickle.dump(lang_embs, file)


for mode in ["cell", "hidden"]:
    for context_mode in ["context", "uncontext"]:
        for layer_mode in ["1", "2"]:
            langvec = {
                lang: lang_rnn_vec[lang][mode][context_mode+layer_mode] for lang in lang_rnn_vec}
            with open(os.path.join(args.out, f"rnnvec_{mode}_{context_mode}{layer_mode}.pkl"), "wb") as file:
                pickle.dump(langvec, file)
