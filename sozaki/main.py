import sys
import torch

from lang_embed import WalsInfo, EmbeddingPairBuilder

with open("wals/codes.csv", "r") as codes_csvfile:
    with open("wals/languages.csv", "r") as languages_csvfile:
        with open("wals/parameters.csv", "r") as parameters_csvfile:
            walsinfo = WalsInfo.from_files(codes_csvfile, languages_csvfile, parameters_csvfile)

with open("sigtyp/dev.csv", "r") as f:
    epb = EmbeddingPairBuilder.from_st2020(f, walsinfo)