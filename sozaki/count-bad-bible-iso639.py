import os
import sys

from onmt.modules.walsinfo import WalsInfo

path = sys.argv[1]
dirs = os.listdir(path)

wals_dir = sys.argv[2]
codes_path      = os.path.join(wals_dir, "codes.csv")
languages_path  = os.path.join(wals_dir, "languages.csv")
parameters_path = os.path.join(wals_dir, "parameters.csv")

with open(codes_path, "r") as codes_csvfile:
    with open(languages_path, "r") as languages_csvfile:
        with open(parameters_path, "r") as parameters_csvfile:
            walsinfo = WalsInfo.from_files(codes_csvfile, languages_csvfile, parameters_csvfile)

iso639p3s = list(walsinfo.wals_code_to_iso639p3.values())

okdirs = []

for dirn in dirs:
    items = dirn.split("_")
    src = items[0]
    tgt = items[1]

    """
    if src not in iso639p3s:
        print("src {} not found in walsinfo (dir = {})".format(src, os.path.join(path, dirn)))

    if tgt not in iso639p3s:
        print("tgt {} not found in walsinfo (dir = {})".format(tgt, os.path.join(path, dirn)))
    """

    if (src in iso639p3s) and (tgt in iso639p3s):
        okdirs.append("\"{}-{}\"".format(src, tgt))

print("OK_DIRS = [ {} ]\n".format(", ".join(okdirs)))
print("OK_DIRS_100 = [ {} ]\n".format(", ".join(okdirs[:100])))