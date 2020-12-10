import sys
import torch

from lang_embed import WalsInfo, EmbeddingPairBuilder

with open("codes.csv", "r") as codes_csvfile:
    with open("languages.csv", "r") as languages_csvfile:
        with open("parameters.csv", "r") as parameters_csvfile:
            walsinfo = WalsInfo.from_files(codes_csvfile, languages_csvfile, parameters_csvfile)

with open("dev.csv", "r") as f:
    epb = EmbeddingPairBuilder.from_st2020(f, walsinfo)