from collections import namedtuple
import csv

import torch
import torch.nn as nn

class WalsInfo:
    def __init__(self):
        ### Maps between WALS code and ISO 639-3.
        ### e.g. "aar" <-> "aiw". Sometimes the ISO 639-3 code is missing,
        ### which would just be the empty string.
        self.wals_code_to_iso639p3 = dict()
        self.iso639p3_to_wals_code = dict()

        ### Maps between paramater index and parameter ID.
        ### e.g. 0 <-> "1A".
        self.param_idx_to_id = []
        self.param_id_to_idx = dict()

        ### Maps between paramater index and parameter name.
        ### e.g. 0 <-> "Consonant Inventories".
        self.param_idx_to_name = []
        self.param_name_to_idx = dict()

        ### Maps parameter index to a map that maps parameter value index to
        ### parameter value name.
        ### e.g. 0 -> {
        ###             1 -> "Small"
        ###             2 -> "Moderately small"
        ###             3 -> "Average"
        ###             4 -> "Moderately Large"
        ###             5 -> "Large"
        ###           }
        self.param_idx_to_param_value_idx_to_param_value_name = dict()

        ### Maps parameter index to a map that maps parameter value name to
        ### parameter value index.
        ### e.g. 0 -> {
        ###             "Small"            -> 1
        ###             "Moderately small" -> 2
        ###             "Average"          -> 3
        ###             "Moderately Large" -> 4
        ###             "Large"            -> 5
        ###           }
        self.param_idx_to_param_value_name_to_param_value_idx = dict()

    def from_files(codes_csvfile, languages_csvfile, parameters_csvfile):
        """
        Get the CSV files from https://github.com/cldf-datasets/wals/blob/master/cldf.
        """

        walsinfo = WalsInfo()

        ### Make the maps between WALS code and ISO 639-3.
        languages_reader = csv.reader(languages_csvfile)

        first_row = True
        for row in languages_reader:
            ### Skip the first row.
            if first_row:
                first_row = False
                continue

            wals_code = row[0]
            iso639p3 = row[6]

            walsinfo.wals_code_to_iso639p3[wals_code] = iso639p3
            walsinfo.iso639p3_to_wals_code[iso639p3] = wals_code

        ### Make the parameters.
        parameters_reader = csv.reader(parameters_csvfile)

        first_row = True
        param_idx = 0
        for row in parameters_reader:
            ### Skip the first row.
            if first_row:
                first_row = False
                continue

            param_id = row[0]
            param_name = row[1]

            walsinfo.param_idx_to_id.append(param_id)
            walsinfo.param_id_to_idx[param_id] = param_idx

            walsinfo.param_idx_to_name.append(param_name)
            walsinfo.param_name_to_idx[param_name] = param_idx
            
            param_idx += 1

        ### Make the parameter values.
        codes_reader = csv.reader(codes_csvfile)

        first_row = True
        param_idx = 0
        param_value_idx_to_param_value_name = dict()
        param_value_name_to_param_value_idx = dict()
        for row in codes_reader:
            ### Skip the first row.
            if first_row:
                first_row = False
                continue

            param_value_id = row[0]
            param_value_name = row[2]

            ### e.g. "1A-1" -> "1A", "1"
            param_id = param_value_id.split("-")[0]
            param_value_idx = param_value_id.split("-")[1]

            ### Are we looking a new parameter?
            if walsinfo.param_id_to_idx[param_id] != param_idx:
                ### Add the dicts and clear them.
                walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx] = param_value_idx_to_param_value_name
                walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx] = param_value_name_to_param_value_idx

                param_value_idx_to_param_value_name = dict()
                param_value_name_to_param_value_idx = dict()

                param_idx += 1
                assert(walsinfo.param_id_to_idx[param_id] == param_idx)

            param_value_idx_to_param_value_name[param_value_idx] = param_value_name
            param_value_name_to_param_value_idx[param_value_name] = param_value_idx

        ### Add the dicts one last time.
        walsinfo.param_idx_to_param_value_idx_to_param_value_name[param_idx] = param_value_idx_to_param_value_name
        walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx] = param_value_name_to_param_value_idx

        return walsinfo

IndexInfo = namedtuple("IndexInfo", [
    "iso639p3", "name", "genus", "family", "param_dict"
])

StringInfo = namedtuple("StringInfo", [
    "iso639p3", "name", "genus", "family", "param_dict"
])

class EmbeddingPairBuilder:
    """
    An object that can be used to construct a pair of special embedding layers, namely a language embedding layer and a language property embedding layer.
    """

    def __init__(self):
        self.iso639p3s = []
        self.names = []
        self.genus_idxs = []
        self.family_idxs = []

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

        ### To be set later.
        self.walsinfo = None

    def from_st2020(csvfile, walsinfo):
        """
        Return a embedding pair builder object constructed from the given `csvfile`.
        
        The specified CSV file must be downloaded from https://github.com/sigtyp/ST2020/blob/master/data.
        """
        
        epb = EmbeddingPairBuilder()
        epb.walsinfo = walsinfo

        reader = csv.reader(csvfile, delimiter="\t")

        first_row = True
        for row in reader:
            ### Skip the first row.
            if first_row:
                first_row = False
                continue

            wals_code = row[0]
            name = row[1]
            genus_string = row[4]
            family_string = row[5]
            params_raw = row[7]

            ### Get the ISO 639-3 code.
            try:
                iso639p3 = walsinfo.wals_code_to_iso639p3[wals_code]
            except KeyError:
                ### We can't even find the ISO 639-3 code, skip
                continue

            ### Get the genus index, adding it into the list/dict if new.
            if genus_string not in epb.genus_i2s:
                genus_idx = len(epb.genus_i2s)
                epb.genus_i2s.append(genus_string)
                epb.genus_s2i[genus_string] = genus_idx

            genus_idx = epb.genus_s2i[genus_string]

            ### Get the family index, adding it into the list/dict if new.
            if family_string not in epb.family_i2s:
                family_idx = len(epb.family_i2s)
                epb.family_i2s.append(family_string)
                epb.family_s2i[family_string] = family_idx

            family_idx = epb.family_s2i[family_string]

            ### Parse the raw param description.
            param_dict = epb.parse_raw_params(params_raw)

            epb.iso639p3s.append(iso639p3)
            epb.names.append(name)
            epb.genus_idxs.append(genus_idx)
            epb.family_idxs.append(family_idx)
            epb.param_dicts.append(param_dict)

        return epb

    def parse_raw_params(self, params_raw):
        """
        Parse raw params, i.e. a string in the 8th column of sigtyp/ST2020 CSV files. Update the feature space list/dict embedding pair builder if necessary.
        """
        param_dict = dict()

        kv_pair_raws = params_raw.split("|")
        for kv_pair_raw in kv_pair_raws:
            split_idx = kv_pair_raw.index("=")
            kv_pair = (
                kv_pair_raw[:split_idx],
                kv_pair_raw[split_idx+1:]
            )

            param_string = kv_pair[0].replace("_", " ")
            param_value_string = kv_pair[1][kv_pair[1].index(" ")+1:]

            ### Get the param index.
            param_idx = self.walsinfo.param_name_to_idx[param_string]

            ### Get the param value index.
            try:
                param_value_idx = self.walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx][param_value_string]
            except KeyError:
                print("param_string:", param_string)
                print("param_idx:", param_idx)
                print("param_value_string:", param_value_string)
                print("param_value_name_to_param_value_idx:", self.walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx])

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

    def make_pair(self):
        """
        """
        
        language_embedding = LanguageEmbedding.from_epb(self)

        return (language_embedding,)

    def print_info(self):
        """
        Print the information encoded in this embedding pair builder.
        """
        
        print("Number of genuses:", len(self.genus_i2s))
        print("Number of families:", len(self.family_i2s))
        print("Number of feature keys:", len(self.feature_key_i2s))
        print("Number of feature values per feature key:")

        hist = []
        
        for key, value_dict in enumerate(self.feature_value_i2d):
            value_dict_size = len(value_dict)

            print("    For feature key \"{}\" (index {}): {}".format(
                self.feature_key_i2s[key], key, value_dict_size
            ))

            if len(hist) < value_dict_size - 1:
                hist.extend([ 0 for _ in range(len(hist), value_dict_size + 1) ])
            
            hist[value_dict_size] += 1

        print("Histogram:")

        for key in range(1, len(hist)):
            print("Number of feature keys with {} feature values: {}".format(
                key, hist[key]
            ))


class LanguageEmbedding(nn.Module):
    """
    An embedding layer that maps a language index into a property lookup tensor, which can then be passed to a property embedding layer to obtain the language embedding for said language.
    """

    def __init__(self):
        pass

    def from_epb(epb):
        """
        Create a language embedding from the specified embedding pair builder `epb`.  The returned language embedding maps a language index to a corresponding property lookup tensor.
        
        Language indices are ordered according to the information stored in `epb`. For example, if Swedish is the 100th language in `epb`, then this language embedding returns the property lookup tensor for Swedish when it receives 100 as its input.

        For the order of the properties, see the documentation for `PropertyEmbedding.from_epb`.
        """

        le = LanguageEmbedding()

        ### Number of embeddings = number of languages.
        num_embeddings = len(epb.wals_codes)

        ### Embedding dimension = sum of the number of values for each feature key over all feature keys + the number of genuses + the number of families.
        embedding_dim = 0

        num_genuses = len(epb.genus_i2s)
        embedding_dim += num_genuses

        num_families = len(epb.family_i2s)
        embedding_dim += num_families

        for value_dict in epb.feature_value_i2d:
            embedding_dim += len(value_dict)

        ### Initialize embeddings. This is the weight tensor of the `nn.Embedding` layer that actually gets updated during training.
        embeddings = torch.zeros((num_embeddings, embedding_dim))

        ### Initialize the gold mask. This indicates which parts of the embedding weight tensor contains gold values.
        le.gold_mask = torch.zeros((num_embeddings, embedding_dim), dtype=torch.bool)

        ### Gold genus and family are defined for all languages.
        le.gold_mask[:, : num_genuses + num_families] = torch.ones((num_embeddings, num_genuses + num_families), dtype=torch.bool)

        ### Loop through the languages, look up the info for each language, and write it into the embeddings.
        for lang_idx in range(len(epb.wals_codes)):
            ### The first columns in the embedding weights are the genus.
            genus_offset = 0
            genus_idx = epb.genus_idxs[lang_idx]
            embeddings[lang_idx][genus_offset + genus_idx] = 1.0

            ### Then, the family.
            family_offset = num_genuses
            family_idx = epb.family_idxs[lang_idx]
            embeddings[lang_idx][family_offset + family_idx] = 1.0

            ### Then, the features.
            feature_offset = num_genuses + num_families
            feature_dict = epb.feature_dicts[lang_idx]
            for key, value_dict in enumerate(epb.feature_value_i2d):
                num_values = len(value_dict)

                if key in feature_dict:
                    value_idx = feature_dict[key]
                    embeddings[lang_idx][feature_offset + value_idx] = 1.0
                    le.gold_mask[lang_idx][feature_offset : feature_offset + num_values] = torch.ones((num_values,), dtype=torch.bool)
                else:
                    le.gold_mask[lang_idx][feature_offset : feature_offset + num_values] = torch.zeros((num_values,), dtype=torch.bool)

                feature_offset += num_values

        ### Make a copy of `embeddings` and store it as the gold embeddings. This contains the gold values and should be used in combination with the gold mask, defined later.
        le.gold_embeddings = embeddings.clone()

        le.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=False
        )

    def forward(self, x):
        return self.embedding(x)

    def overwrite_gold(self):
        not_gold_mask = self.gold_mask.logical_not()

        self.embedding.weight = self.gold_mask * self.gold_embeddings + not_gold_mask * self.embedding.weight