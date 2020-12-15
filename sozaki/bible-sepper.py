import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(
    description="Adds language ID tokens to the bible dataset.")

parser.add_argument(
    "bible_dir", type=str,
    help="Directory that contains the various src_tgt subdirectories, i.e. /path/to/bible-com."
)

parser.add_argument(
    "out_dir", type=str,
    help="Directory to which the separated data should be written."
)

args = parser.parse_args()

altogether_dir = os.path.join(args.out_dir, "altogether")
os.makedirs(altogether_dir)

altogether_src_path = os.path.join(altogether_dir, "src.txt")
altogether_tgt_path = os.path.join(altogether_dir, "tgt.txt")
altogether_dedup_tgt_path = os.path.join(altogether_dir, "dedup-tgt.txt")

dedup_tgt_sentences = set()
with open(altogether_src_path, "w+") as altogether_src:
    with open(altogether_tgt_path, "w+") as altogether_tgt:
        for subdir in tqdm(os.listdir(args.bible_dir)):
            items = subdir.split("_")
            src = items[0]
            tgt = items[1]

            src_tgt_dir = os.path.join(args.out_dir, "{}-{}".format(src, tgt))
            os.makedirs(src_tgt_dir)

            orig_path = "bible.orig.{}-{}".format(src, tgt)
            orig_path = os.path.join(args.bible_dir, subdir, orig_path)

            src_sents = []
            tgt_sents = []

            with open(orig_path, "r") as f:
                for line in f:
                    items = line.split("|||")
                    src_sent = items[0].strip()
                    tgt_sent = items[1].strip()

                    src_sents.append(src_sent)
                    tgt_sents.append(tgt_sent)
            dedup_tgt_sentences.update(tgt_sents)

            src_path = os.path.join(src_tgt_dir, "src.txt")
            tgt_path = os.path.join(src_tgt_dir, "tgt.txt")

            with open(src_path, "w+") as src:
                src.write("\n".join(src_sents))
                src.write("\n")

            with open(tgt_path, "w+") as tgt:
                tgt.write("\n".join(tgt_sents))
                tgt.write("\n")

            altogether_src.write("\n".join(src_sents))
            altogether_src.write("\n")

            altogether_tgt.write("\n".join(tgt_sents))
            altogether_tgt.write("\n")


# dedup tgt
dedup_tgt_sentences.remove("")
with open(altogether_dedup_tgt_path, "w") as altogether_dedup_tgt:
    altogether_dedup_tgt.write("\n".join(list(dedup_tgt_sentences))+"\n")
