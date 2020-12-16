import argparse
import sentencepiece as spm
from tqdm import tqdm
import os
import random
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description="Performs BPE on the bible dataset.")

parser.add_argument(
    "sep_bible_dir", type=str,
    help="Directory that contains the various src-tgt subdirectories, i.e. /path/to/sep-bible-com, i.e. the output directory of bible-sepper.py"
)

parser.add_argument(
    "out_dir", type=str,
    help="Directory to which the BPE'ed data should be written."
)

args = parser.parse_args()

altogether_dir = os.path.join(args.sep_bible_dir, "altogether")

altogether_src_path = os.path.join(altogether_dir, "src.txt")
altogether_tgt_path = os.path.join(altogether_dir, "dedup-tgt.txt")

print("Training BPE on {}+{}...".format(altogether_src_path, altogether_tgt_path))

if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

model_prefix = os.path.join(args.out_dir, "model")
print("Model will be written to {}.{{ model, vocab }}.".format(model_prefix))

if not os.path.isfile(model_prefix+".model") or not os.path.isfile(model_prefix+".vocab"):
    spm.SentencePieceTrainer.train(
        input="{},{}".format(altogether_src_path, altogether_tgt_path),
        model_prefix=model_prefix,
        vocab_size=64000,
        character_coverage=0.9995,
        train_extremely_large_corpus=True,
        num_threads=128
    )

print("Encoding files from {} with the model...".format(args.sep_bible_dir))

"""
e.g. sep-bible-com/deu-eng -->

out-dir/deu-eng
 |--eng-as-src-nolang
 |   |--src.txt
 |   \--tgt.txt
 |--eng-as-tgt-nolang
 |   |--src.txt
 |   \--tgt.txt
 |--eng-as-src-lang
 |   |--src.txt
 |   \--tgt.txt
 \--eng-as-tgt-lang
     |--src.txt
     \--tgt.txt

also:

out-dir/altogether
 |--eng-as-src-nolang
 |   |--src.txt
 |   \--tgt.txt
 |--eng-as-tgt-nolang
 |   |--src.txt
 |   \--tgt.txt
 |--eng-as-src-lang
 |   |--src.txt
 |   \--tgt.txt
 \--eng-as-tgt-lang
     |--src.txt
     \--tgt.txt
"""

model_path = model_prefix + ".model"
sp = spm.SentencePieceProcessor(model_file=model_path)

altogether_out_files = dict()
altogether_out_dir = os.path.join(args.out_dir, "altogether")
if not os.path.isdir(altogether_out_dir):
    os.mkdir(altogether_out_dir)

for engside in ("eng-as-src", "eng-as-tgt"):
    altogether_out_files[engside] = dict()

    for lang in ("lang", "nolang"):
        altogether_out_files[engside][lang] = dict()
        if not os.path.isdir(os.path.join(altogether_out_dir, "{}-{}".format(engside, lang))):
            os.mkdir(os.path.join(altogether_out_dir,
                                  "{}-{}".format(engside, lang)))
        for side in ("src", "tgt", "src-train", "src-val", "tgt-train", "tgt-val", "src-val-sampled", "tgt-val-sampled"):
            path = os.path.join(altogether_out_dir,
                                "{}-{}".format(engside, lang), "{}.txt".format(side))
            altogether_out_files[engside][lang][side] = open(path, "w+")

special_tokens = set()
RATIO_VAL = 0.1
for subdir in tqdm(os.listdir(args.sep_bible_dir)):
    if subdir == "altogether":
        continue
    subdir_out_dir = os.path.join(args.out_dir, subdir)
    if not os.path.isdir(subdir_out_dir):
        os.mkdir(subdir_out_dir)
    items = subdir.split("-")
    noneng = items[0]

    noneng_lang = "LANG_" + noneng.upper()
    special_tokens.add(noneng_lang)

    src_path = os.path.join(args.sep_bible_dir, subdir, "src.txt")
    tgt_path = os.path.join(args.sep_bible_dir, subdir, "tgt.txt")

    noneng_out = []
    noneng_out_lang = []
    eng_out = []
    eng_out_lang = []

    with open(src_path, "r") as srcf:
        for line in srcf:
            encoded = " ".join(sp.encode(line.strip(), out_type=str))
            noneng_out.append(encoded)
            noneng_out_lang.append(noneng_lang + " " + encoded)

    with open(tgt_path, "r") as tgtf:
        for line in tgtf:
            encoded = " ".join(sp.encode(line.strip(), out_type=str))
            eng_out.append(encoded)
            eng_out_lang.append(noneng_lang + " " + encoded)

    dict_noneng_out = {"full": noneng_out}
    dict_noneng_out_lang = {"full": noneng_out_lang}
    dict_eng_out = {"full": eng_out}
    dict_eng_out_lang = {"full": eng_out_lang}
    try:
        temp_train, temp_val = train_test_split(
            list(zip(noneng_out, noneng_out_lang, eng_out, eng_out_lang)), test_size=RATIO_VAL)
        sampled_temp_val = random.sample(temp_val, min(1000, len(temp_val)))
        dict_noneng_out["train"], dict_noneng_out_lang["train"], dict_eng_out["train"], dict_eng_out_lang["train"] = zip(
            *temp_train)
        dict_noneng_out["val"], dict_noneng_out_lang["val"], dict_eng_out["val"], dict_eng_out_lang["val"] = zip(
            *temp_val)
        dict_noneng_out["val-sampled"], dict_noneng_out_lang["val-sampled"], dict_eng_out["val-sampled"], dict_eng_out_lang["val-sampled"] = zip(
            *sampled_temp_val)
    except:
        print(subdir)
        print(len(noneng_out))
        print(len(noneng_out_lang))
        print(len(eng_out))
        print(len(eng_out_lang))
        continue

    def pack_and_write(lines, f1, f2):
        content = "\n".join(lines) + "\n"
        f1.write(content)
        f2.write(content)

    for engside in ("eng-as-src", "eng-as-tgt"):
        for lang in ("lang", "nolang"):
            for side in ("src", "tgt"):
                subdir_out_lang_dir = os.path.join(
                    subdir_out_dir, "{}-{}".format(engside, lang))
                if not os.path.isdir(subdir_out_lang_dir):
                    os.mkdir(subdir_out_lang_dir)
                path = os.path.join(subdir_out_lang_dir, "{}.txt".format(side))
                path_train = os.path.join(
                    subdir_out_lang_dir, "{}-train.txt".format(side))
                path_val = os.path.join(
                    subdir_out_lang_dir, "{}-val.txt".format(side))
                path_val_sample = os.path.join(
                    subdir_out_lang_dir, "{}-val-sampled.txt".format(side))

                with open(path, "w+") as f:
                    if (engside, lang, side) == ("eng-as-src", "lang", "src"):
                        pack_and_write(
                            eng_out_lang, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_eng_out_lang["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_eng_out_lang["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_eng_out_lang["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-src", "lang", "tgt"):
                        pack_and_write(
                            noneng_out, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_noneng_out["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_noneng_out["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_noneng_out["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-src", "nolang", "src"):
                        pack_and_write(
                            eng_out, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_eng_out["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_eng_out["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_eng_out["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-src", "nolang", "tgt"):
                        pack_and_write(
                            noneng_out, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_noneng_out["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_noneng_out["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_noneng_out["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-tgt", "lang", "src"):
                        pack_and_write(
                            noneng_out_lang, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_noneng_out_lang["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_noneng_out_lang["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_noneng_out_lang["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-tgt", "lang", "tgt"):
                        pack_and_write(
                            eng_out, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_eng_out["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_eng_out["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_eng_out["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-tgt", "nolang", "src"):
                        pack_and_write(
                            noneng_out, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_noneng_out["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_noneng_out["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_noneng_out["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

                    elif (engside, lang, side) == ("eng-as-tgt", "nolang", "tgt"):
                        pack_and_write(
                            eng_out, f, altogether_out_files[engside][lang][side])
                        with open(path_train, "w+") as f_train:
                            pack_and_write(
                                dict_eng_out["train"], f_train, altogether_out_files[engside][lang][side+"-train"])
                        with open(path_val, "w+") as f_val:
                            pack_and_write(
                                dict_eng_out["val"], f_val, altogether_out_files[engside][lang][side+"-val"])
                        with open(path_val_sample, "w+") as f_val_sample:
                            pack_and_write(
                                dict_eng_out["val-sampled"], f_val_sample, altogether_out_files[engside][lang][side+"-val-sampled"])

#   Close the files.
for engside in ("eng-as-src", "eng-as-tgt"):
    for lang in ("lang", "nolang"):
        for side in ("src", "tgt", "src-train", "src-val", "tgt-train", "tgt-val", "src-val-sampled", "tgt-val-sampled"):
            altogether_out_files[engside][lang][side].close()

# save special language tokens
with open(os.path.join(args.out_dir, "model.vocab.special"), "w") as file:
    file.write("\n".join(list(special_tokens))+"\n")
