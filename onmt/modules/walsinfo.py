import csv
from onmt.utils.logging import logger

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
            param_value_idx = int(param_value_id.split("-")[1])

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