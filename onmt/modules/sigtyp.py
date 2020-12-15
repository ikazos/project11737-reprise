from collections import namedtuple
import csv
from onmt.utils.logging import logger

from onmt.modules.special_embeddings import LookupEmbedding

IndexInfo = namedtuple("IndexInfo", [
    "iso639p3", "name", "genus", "family", "param_dict"
])

StringInfo = namedtuple("StringInfo", [
    "iso639p3", "name", "genus", "family", "param_dict"
])

class Sigtyp:
    """
    An object that can be used to construct a pair of special embedding layers, namely a language embedding layer and a language property embedding layer.
    """

    def __init__(self, walsinfo, use_all_params=False):
        self.iso639p3s = []
        self.names = []
        self.genus_idxs = []
        self.family_idxs = []

        self.walsinfo = walsinfo

        ### Each dictionary maps param index to param value index.
        self.param_dicts = []

        ### Maps genus index to genus (string).
        self.genus_i2s = []

        ### Maps genus (string) to genus index.
        self.genus_s2i = dict()

        ### Maps family index to family (string).
        self.family_i2s = []

        ### Maps family (string) to family index.
        self.family_s2i = dict()

        ### The set of param indices that are actually used in this dataset.
        self.used_param_idxs = set()
        if use_all_params:
            self.used_param_idxs = set(range(len(self.walsinfo.param_idx_to_id)))

    def add_from_st2020(self, csvfile):
        """
        Return a embedding pair builder object constructed from the given `csvfile`.
        
        The specified CSV file must be downloaded from https://github.com/sigtyp/ST2020/blob/master/data.
        """

        reader = csv.reader(csvfile, delimiter="\t")

        first_row = True
        for row in reader:
            ### Skip the first row.
            if first_row:
                first_row = False
                continue

            if len(row) > 8:
                for k, item in enumerate(row):
                    print("Item {}: {}".format(k, item))
                raise Exception("lol")

            wals_code = row[0]
            name = row[1]
            genus_string = row[4]
            family_string = row[5]
            params_raw = row[7]

            ### Get the ISO 639-3 code.
            try:
                iso639p3 = self.walsinfo.wals_code_to_iso639p3[wals_code]
            except KeyError:
                ### We can't even find the ISO 639-3 code, skip
                continue

            ### Get the genus index, adding it into the list/dict if new.
            if genus_string not in self.genus_i2s:
                genus_idx = len(self.genus_i2s)
                self.genus_i2s.append(genus_string)
                self.genus_s2i[genus_string] = genus_idx

            genus_idx = self.genus_s2i[genus_string]

            ### Get the family index, adding it into the list/dict if new.
            if family_string not in self.family_i2s:
                family_idx = len(self.family_i2s)
                self.family_i2s.append(family_string)
                self.family_s2i[family_string] = family_idx

            family_idx = self.family_s2i[family_string]

            ### Parse the raw param description.
            param_dict = self.parse_raw_params(params_raw)

            self.iso639p3s.append(iso639p3)
            self.names.append(name)
            self.genus_idxs.append(genus_idx)
            self.family_idxs.append(family_idx)
            self.param_dicts.append(param_dict)

    def parse_raw_params(self, params_raw):
        """
        Parse raw params, i.e. a string in the 8th column of sigtyp/ST2020 CSV files. Update the feature space list/dict embedding pair builder if necessary.
        """
        param_dict = dict()

        ### params_raw:
        ### Consonant_Inventories=1 Small|Vowel_Quality_Inventories=2 Average (5-6)|...

        kv_pair_raws = params_raw.split("|")
        for kv_pair_raw in kv_pair_raws:
            split_idx = kv_pair_raw.index("=")
            kv_pair = (
                kv_pair_raw[:split_idx],            ### Consonant_Inventories
                kv_pair_raw[split_idx+1:]           ### 1 Small
            )

            param_string = kv_pair[0].replace("_", " ")                 ### Consonant Inventories

            ### Get the param index.
            param_idx = self.walsinfo.param_name_to_idx[param_string]

            self.used_param_idxs.add(param_idx)

            ### Only try and find the param value if it's not "?", i.e. unknown.
            if kv_pair[1] != "?":
                param_value_string = kv_pair[1][kv_pair[1].index(" ")+1:]   ### Small

                ### Get the param value index.
                try:
                    param_value_idx = self.walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx][param_value_string]
                except KeyError as e:
                    print("param_string:", param_string)
                    print("param_idx:", param_idx)
                    print("param_value_string:", param_value_string)
                    print("param_value_name_to_param_value_idx:", self.walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx])
                    print("params_raw:", params_raw)

                    raise e

                param_dict[param_idx] = param_value_idx

        return param_dict

    def get_index_info(self, lang_idx):
        """
        Get IndexInfo for a language with index `lang_idx`.
        """
        return IndexInfo(
            self.iso639p3s[lang_idx],
            self.names[lang_idx],
            self.genus_idxs[lang_idx],
            self.family_idxs[lang_idx],
            self.param_dicts[lang_idx]
        )

    def get_string_info(self, lang_idx):
        """
        Get StringInfo for a language with index `lang_idx`.
        """

        genus = self.genus_i2s[self.genus_idxs[lang_idx]]
        family = self.family_i2s[self.family_idxs[lang_idx]]

        param_dict = self.param_dicts[lang_idx]
        param_string_dict = dict()

        for param_idx, param_value_idx in param_dict.items():
            param_name = self.walsinfo.param_idx_to_name[param_idx]
            param_value_name = self.walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx][param_value_idx]

            param_string_dict[param_name] = param_value_name

        return StringInfo (
            self.iso639p3s[lang_idx],
            self.names[lang_idx],
            genus,
            family,
            param_string_dict
        )

    def print_info(self):
        """
        Print the information encoded in this embedding pair builder.
        """
        
        print("Number of genuses:", len(self.genus_i2s))
        print("Number of families:", len(self.family_i2s))
        print("Number of params:", len(self.used_param_idxs))