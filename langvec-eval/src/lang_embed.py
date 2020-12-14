from collections import namedtuple
import csv

# import torch
# import torch.nn as nn


class WalsInfo:
    def __init__(self):
        # Maps between WALS code and ISO 639-3.
        # e.g. "aar" <-> "aiw". Sometimes the ISO 639-3 code is missing,
        # which would just be the empty string.
        self.wals_code_to_iso639p3 = dict()
        self.iso639p3_to_wals_code = dict()

        # Maps between paramater index and parameter ID.
        # e.g. 0 <-> "1A".
        self.param_idx_to_id = []
        self.param_id_to_idx = dict()

        # Maps between paramater index and parameter name.
        # e.g. 0 <-> "Consonant Inventories".
        self.param_idx_to_name = []
        self.param_name_to_idx = dict()

        # Maps parameter index to a map that maps parameter value index to
        # parameter value name.
        # e.g. 0 -> {
        # 1 -> "Small"
        # 2 -> "Moderately small"
        # 3 -> "Average"
        # 4 -> "Moderately Large"
        # 5 -> "Large"
        # }
        self.param_idx_to_param_value_idx_to_param_value_name = dict()

        # Maps parameter index to a map that maps parameter value name to
        # parameter value index.
        # e.g. 0 -> {
        # "Small"            -> 1
        # "Moderately small" -> 2
        # "Average"          -> 3
        # "Moderately Large" -> 4
        # "Large"            -> 5
        # }
        self.param_idx_to_param_value_name_to_param_value_idx = dict()

    def from_files(codes_csvfile, languages_csvfile, parameters_csvfile):
        """
        Get the CSV files from https://github.com/cldf-datasets/wals/blob/master/cldf.
        """

        walsinfo = WalsInfo()

        # Make the maps between WALS code and ISO 639-3.
        languages_reader = csv.reader(languages_csvfile)

        first_row = True
        for row in languages_reader:
            # Skip the first row.
            if first_row:
                first_row = False
                continue

            wals_code = row[0]
            iso639p3 = row[6]

            walsinfo.wals_code_to_iso639p3[wals_code] = iso639p3
            walsinfo.iso639p3_to_wals_code[iso639p3] = wals_code

        # Make the parameters.
        parameters_reader = csv.reader(parameters_csvfile)

        first_row = True
        param_idx = 0
        for row in parameters_reader:
            # Skip the first row.
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

        # Make the parameter values.
        codes_reader = csv.reader(codes_csvfile)

        first_row = True
        param_idx = 0
        param_value_idx_to_param_value_name = dict()
        param_value_name_to_param_value_idx = dict()
        for row in codes_reader:
            # Skip the first row.
            if first_row:
                first_row = False
                continue

            param_value_id = row[0]
            param_value_name = row[2]

            # e.g. "1A-1" -> "1A", "1"
            param_id = param_value_id.split("-")[0]
            param_value_idx = param_value_id.split("-")[1]

            # Are we looking a new parameter?
            if walsinfo.param_id_to_idx[param_id] != param_idx:
                # Add the dicts and clear them.
                walsinfo.param_idx_to_param_value_idx_to_param_value_name[
                    param_idx] = param_value_idx_to_param_value_name
                walsinfo.param_idx_to_param_value_name_to_param_value_idx[
                    param_idx] = param_value_name_to_param_value_idx

                param_value_idx_to_param_value_name = dict()
                param_value_name_to_param_value_idx = dict()

                param_idx += 1
                assert(walsinfo.param_id_to_idx[param_id] == param_idx)

            param_value_idx_to_param_value_name[param_value_idx] = param_value_name
            param_value_name_to_param_value_idx[param_value_name] = param_value_idx

        # Add the dicts one last time.
        walsinfo.param_idx_to_param_value_idx_to_param_value_name[
            param_idx] = param_value_idx_to_param_value_name
        walsinfo.param_idx_to_param_value_name_to_param_value_idx[
            param_idx] = param_value_name_to_param_value_idx

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

        # Each dictionary maps param index to param value index.
        self.param_dicts = []

        # Each dictionary maps blinded param index to param value index.
        self.blinded_param_dicts = []

        # Maps genus index to genus (string).
        self.genus_i2s = []

        # Maps genus (string) to genus index.
        self.genus_s2i = dict()

        # Maps family index to family (string).
        self.family_i2s = []

        # Maps family (string) to family index.
        self.family_s2i = dict()

        # To be set later.
        self.walsinfo = None

    def from_st2020(csvfile, walsinfo, blinded=False):
        """
        Return a embedding pair builder object constructed from the given `csvfile`.

        The specified CSV file must be downloaded from https://github.com/sigtyp/ST2020/blob/master/data.
        """

        epb = EmbeddingPairBuilder()
        epb.walsinfo = walsinfo

        reader = csv.reader(csvfile, delimiter="\t")

        first_row = True
        for row in reader:
            # Skip the first row.
            if first_row:
                first_row = False
                continue

            wals_code = row[0]
            name = row[1]
            genus_string = row[4]
            family_string = row[5]
            params_raw = row[7]
            if len(row) == 9:
                params_raw += "  "+row[8]

            # Get the ISO 639-3 code.
            try:
                iso639p3 = walsinfo.wals_code_to_iso639p3[wals_code]
            except KeyError:
                # We can't even find the ISO 639-3 code, skip
                continue

            # Get the genus index, adding it into the list/dict if new.
            if genus_string not in epb.genus_i2s:
                genus_idx = len(epb.genus_i2s)
                epb.genus_i2s.append(genus_string)
                epb.genus_s2i[genus_string] = genus_idx

            genus_idx = epb.genus_s2i[genus_string]

            # Get the family index, adding it into the list/dict if new.
            if family_string not in epb.family_i2s:
                family_idx = len(epb.family_i2s)
                epb.family_i2s.append(family_string)
                epb.family_s2i[family_string] = family_idx

            family_idx = epb.family_s2i[family_string]

            # Parse the raw param description.
            param_dict, blinded_param_dict = epb.parse_raw_params(
                params_raw, blinded)

            epb.iso639p3s.append(iso639p3)
            epb.names.append(name)
            epb.genus_idxs.append(genus_idx)
            epb.family_idxs.append(family_idx)
            epb.param_dicts.append(param_dict)
            epb.blinded_param_dicts.append(blinded_param_dict)

        return epb

    def parse_raw_params(self, params_raw, blinded=False):
        """
        Parse raw params, i.e. a string in the 8th column of sigtyp/ST2020 CSV files. Update the feature space list/dict embedding pair builder if necessary.
        """
        param_dict = dict()
        blinded_param_dict = dict()

        kv_pair_raws = params_raw.split("|")
        for kv_pair_raw in kv_pair_raws:
            split_idx = kv_pair_raw.index("=")
            kv_pair = (
                kv_pair_raw[:split_idx],
                kv_pair_raw[split_idx+1:]
            )

            param_string = kv_pair[0].replace("_", " ")
            if " " in kv_pair[1]:
                param_value_string = kv_pair[1][kv_pair[1].index(" ")+1:]
            else:
                param_value_string = kv_pair[1].strip()

            # Get the param index.
            param_idx = self.walsinfo.param_name_to_idx[param_string]

            # Get the param value index.
            try:
                param_value_idx = self.walsinfo.param_idx_to_param_value_name_to_param_value_idx[
                    param_idx][param_value_string]
            except KeyError:
                if param_value_string == "?":
                    param_value_idx = -1
                    blinded_param_dict[param_idx] = -1
                else:
                    print("param_string:", param_string)
                    print("param_idx:", param_idx)
                    print("param_value_string:", param_value_string)
                    print("param_value_name_to_param_value_idx:",
                          self.walsinfo.param_idx_to_param_value_name_to_param_value_idx[param_idx])
                continue

            param_dict[param_idx] = param_value_idx

        return param_dict, blinded_param_dict

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
            param_value_name = self.walsinfo.param_idx_to_param_value_idx_to_param_value_name[
                param_idx][param_value_idx]

            param_string_dict[param_name] = param_value_name

        return StringInfo(
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
                hist.extend([0 for _ in range(len(hist), value_dict_size + 1)])

            hist[value_dict_size] += 1

        print("Histogram:")

        for key in range(1, len(hist)):
            print("Number of feature keys with {} feature values: {}".format(
                key, hist[key]
            ))
