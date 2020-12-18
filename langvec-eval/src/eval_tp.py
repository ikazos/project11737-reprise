import os
import pickle
import numpy as np
import argparse
from os.path import join
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from utils import load_langvec, save_pkl, save_json
from config import PATH_DATA_TYP, PATH_DATA_WALS, PATH_SAVE_MODEL, PATH_DATA_LANGVEC
from lang_embed import WalsInfo, EmbeddingPairBuilder

import warnings
warnings.filterwarnings('ignore')

PATH_BASELINE_MT = join(PATH_DATA_LANGVEC, "mtvec.pkl")
PATH_BASELINE_MTCELL = join(PATH_DATA_LANGVEC, "mtcell.pkl")
PATH_BASELINE_MTCELL2 = join(PATH_DATA_LANGVEC, "mtcell2.pkl")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--langvec", "-lv", default=None, required=False, type=str,
                    help="path to the langvec dir")
parser.add_argument("--exclude_baseline", "-eb",
                    action='store_true', default=False)
parser.add_argument("--verbose", "-v", default=False, type=bool)
args = parser.parse_args()

# assert os.path.isdir(args.langvec), f"{args.langvec} is not a directory"


def get_name_from_path(path):
    name = path.split("/")[-1]
    name = ".".join(name.split(".")[:-1])
    return name


def get_path_langvecs(out_path=None):
    path_langvecs = [PATH_BASELINE_MT,
                     PATH_BASELINE_MTCELL, PATH_BASELINE_MTCELL2] if not args.exclude_baseline else []
    if out_path is not None:
        print(
            f"testing {os.listdir(out_path)} + 3 baselines(mtvec, mtcell, mtcell2)")
        path_langvecs += [join(args.langvec, f)
                          for f in os.listdir(out_path)]
    path_langvecs = [(get_name_from_path(p), p) for p in path_langvecs]
    return path_langvecs


def load_sigtyp_csv(path, walsinfo, blinded=False):
    with open(path, "r") as f:
        epb = EmbeddingPairBuilder.from_st2020(f, walsinfo, blinded)
    return epb


def load_data_tp(path_data=PATH_DATA_TYP, path_wals=PATH_DATA_WALS):
    """load data for typology prediction. use WalsInfo, EmbeddingPairBuilder classes from lang_embed.py"""

    def combine_test(test_blinded, test_gold):
        assert len(test_blinded.param_dicts) == len(test_gold.param_dicts)
        assert test_blinded.names == test_gold.names
        test_gold.blinded_param_dicts = test_blinded.blinded_param_dicts
        return test_gold

    with open(os.path.join(path_wals, "codes.csv"), "r") as codes_csvfile:
        with open(os.path.join(path_wals, "languages.csv"), "r") as languages_csvfile:
            with open(os.path.join(path_wals, "parameters.csv"), "r") as parameters_csvfile:
                walsinfo = WalsInfo.from_files(
                    codes_csvfile, languages_csvfile, parameters_csvfile)
    train = load_sigtyp_csv(os.path.join(path_data, "train.csv"), walsinfo)
    dev = load_sigtyp_csv(os.path.join(path_data, "dev.csv"), walsinfo)
    test_blinded = load_sigtyp_csv(os.path.join(
        path_data, "test_blinded.csv"), walsinfo, blinded=True)
    test_gold = load_sigtyp_csv(os.path.join(
        path_data, "test_gold.csv"), walsinfo)

    test = combine_test(test_blinded, test_gold)
    return walsinfo, train, dev, test


def train_tp(vec_name, langvec, walsinfo, train, dev, test=None, path_save=None):
    def construct_xy_blinded(langvec, data, param_idx):
        langs, xs, ys = [], [], []
        for name, param_gold, param_blinded in zip(data.iso639p3s, data.param_dicts, data.blinded_param_dicts):
            if param_idx in param_blinded:
                if name in langvec:
                    langs.append(name)
                    x_lang = langvec[name]
                    if not isinstance(x_lang, list):
                        x_lang = [t.tolist() for t in x_lang]
                    xs.append(x_lang)
                    ys.append(param_gold[param_idx])
                else:
                    # print(f"{name} not in langvec")
                    pass
        return xs, ys, langs

    def construct_xy(langvec, data, param_idx):
        xs, ys = [], []
        for name, param in zip(data.iso639p3s, data.param_dicts):
            if param_idx in param:
                if name in langvec:
                    x_lang = langvec[name]
                    if not isinstance(x_lang, list):
                        x_lang = [t.tolist() for t in x_lang]
                    xs.append(x_lang)
                    ys.append(param[param_idx])
                else:
                    # print(f"{name} not in langvec")
                    pass
        return xs, ys

    def summarize_res(res):
        total_dev_f1, total_dev_acc, total_test_f1, total_test_acc, total_dev, total_test = 0, 0, 0, 0, 0, 0
        for param_idx, res_dict in res.items():
            dev_f1, dev_acc = res_dict['dev']['f1'], res_dict['dev']['acc']
            test_f1, test_acc = res_dict['test']['f1'], res_dict['test']['acc']
            if dev_f1 and dev_acc:
                total_dev_f1 += dev_f1
                total_dev_acc += dev_acc
                total_dev += 1
            if test_f1 and test_acc:
                total_test_f1 += test_f1
                total_test_acc += test_acc
                total_test += 1
        avg_dev_f1 = total_dev_f1/total_dev
        avg_dev_acc = total_dev_acc/total_dev
        avg_test_f1 = total_test_f1/total_test
        avg_test_acc = total_test_acc/total_test
        avg_res = {"dev_f1": avg_dev_f1, "dev_acc": avg_dev_acc,
                   "test_f1": avg_test_f1, "test_acc": avg_test_acc}
        if args.verbose:
            print(
                f"avg. Dev f1: {avg_dev_f1:.3f}\navg. Dev acc: {avg_dev_acc:.3f}\navg. Test f1: {avg_test_f1:.3f}\navg. Test acc: {avg_test_acc:.3f}")
        return avg_res

    def train_one_model(x_train, y_train, x_dev=None, y_dev=None, path_save=None, std=False, use_weight=False, pca=False, use_lin_train=False, categorical=True, random_seed=42, max_iter=2000):
        if x_dev:
            x_train = x_train + x_dev
            y_train = y_train + y_dev

        if std:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            # if x_dev:
            #     x_dev = scaler.transform(x_dev)
        else:
            scaler = None

        if pca:
            pca = PCA(n_components=300)
            x_train = pca.fit_transform(x_train)
            # if x_dev:
            #     x_dev = pca.fit_transform(x_dev)
        solver, c = "lbfgs", 1.0
        model = LogisticRegression(random_state=random_seed, multi_class="multinomial", solver=solver, C=c,
                                   max_iter=max_iter, penalty="l2", class_weight=None)
        # print(type(x_train))
        # print(type(x_train[0]))
        # print(len(x_train))
        # print(len(x_train[0]))
        # print(len(y_train))
        try:
            model.fit(x_train, y_train)
        except:
            print(x_train)
            print(y_train)
            raise
        return model, scaler

    def eval_one_label(pred_label, x_dev, y_dev, x_test=None, y_test=None):
        dev_macro_f1, dev_acc = None, None
        if x_dev != []:
            y_pred = np.array([pred_label for _ in x_dev])
            dev_macro_f1 = classification_report(y_dev, y_pred, output_dict=True)[
                "macro avg"]["f1-score"]
            dev_acc = sum(list(y_dev == y_pred))/len(y_dev)

        test_macro_f1, test_acc, test_y_pred = None, None, None
        if x_test is not None and x_test != []:
            test_y_pred = np.array([pred_label for _ in x_test])
            test_macro_f1 = classification_report(y_test, test_y_pred, output_dict=True)[
                "macro avg"]["f1-score"]
            test_acc = sum(list(y_test == test_y_pred))/len(y_test)

        res = {'dev': {'f1': dev_macro_f1, 'acc': dev_acc},
               'test': {'f1': test_macro_f1, 'acc': test_acc}}
        return res, test_y_pred

    def eval(model, x_dev, y_dev, x_test=None, y_test=None, scaler=None):
        if scaler:
            if x_dev != []:
                x_dev = scaler.transform(x_dev)
            if x_test:
                x_test = scaler.transform(x_test)

        dev_macro_f1, dev_acc = None, None
        if x_dev is not None and x_dev != []:
            y_pred = model.predict(x_dev)
            dev_macro_f1 = classification_report(y_dev, y_pred, output_dict=True)[
                "macro avg"]["f1-score"]
            dev_acc = sum(list(y_dev == y_pred))/len(y_dev)

        test_macro_f1, test_acc, test_y_pred = None, None, None
        if x_test is not None and x_test != []:
            if scaler:
                x_test = scaler.transform(x_test)
            test_y_pred = model.predict(x_test)
            test_macro_f1 = classification_report(y_test, test_y_pred, output_dict=True)[
                "macro avg"]["f1-score"]
            test_acc = sum(list(y_test == test_y_pred))/len(y_test)

        res = {'dev': {'f1': dev_macro_f1, 'acc': dev_acc},
               'test': {'f1': test_macro_f1, 'acc': test_acc}}
        return res, test_y_pred

    models = {}
    models_res = {}
    models_preds = {}
    for param_name, param_idx in tqdm(walsinfo.param_name_to_idx.items(), total=len(walsinfo.param_name_to_idx)):
        x_train, y_train = construct_xy(langvec, train, param_idx)
        x_dev, y_dev = construct_xy(langvec, dev, param_idx)
        if len(x_train) == 0 and len(x_dev) == 0:
            if args.verbose:
                print(f"{param_name} does not exist in  train data")
            continue
        if len(set(y_train+y_dev)) == 1:
            pred_label = list(set(y_train+y_dev))[0]
            res, preds = eval_one_label(
                pred_label, x_dev, y_dev, x_test, y_test)
        else:
            if test != None:
                x_test, y_test, langs_test = construct_xy_blinded(
                    langvec, test, param_idx)
            else:
                x_test, y_test, langs_test = None, None, None

            model, scaler = train_one_model(x_train, y_train, x_dev, y_dev)
            res, preds = eval(model, x_dev, y_dev, x_test, y_test, scaler)

        models[param_idx] = model
        models_res[param_idx] = res
        if langs_test != [] and preds is not None:
            models_preds[param_idx] = list(zip(langs_test, preds))
        else:
            models_preds[param_idx] = None

    avg_res = summarize_res(models_res)
    if path_save:
        save_pkl(models, path_save+"models.pkl")
        save_pkl(models_res, path_save+"res.pkl")
        save_pkl(models_preds, path_save+"preds.pkl")
        save_json(avg_res, path_save+"summary.json")

    return models, models_preds, avg_res


def infer_tp():
    pass


def score_tp():
    pass


if __name__ == "__main__":
    path_langvecs = get_path_langvecs(args.langvec)
    walsinfo, train, dev, test = load_data_tp()

    res = {}
    for vec_name, path in path_langvecs[:2]:
        langvec = load_langvec(path)
        models, models_preds, model_res = train_tp(vec_name, langvec, walsinfo, train, dev, test,
                                                   PATH_SAVE_MODEL+"typology-prediction/")
        res[vec_name] = model_res
