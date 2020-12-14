from config import PATH_DATA_TYP, PATH_DATA_WALS
from utils import load_langvec
from eval_tp import load_data_tp, train_tp, infer_tp, score_tp
import argparse

parser = argparse.ArgumentParser(description='Process arguments for langvec evaluation')

# paths
parser.add_argument('--langvec','-l', type=str, required=True, help="path to langvec pkl file")
parser.add_argument('--data_typology','-dt', type=str, default=PATH_DATA_TYP, help="path to data for typology prediction")
parser.add_argument('--data_wals','-dw', type=str, default=PATH_DATA_WALS, help="path to WALS data")
parser.add_argument('--data_phylogenetic','-dp', type=str, required=True, help="path to data for phylogenetic tree reconstruction")
parser.add_argument('--data_selection','-ds', type=str, required=True, help="path to data for language selection")

# tasks
parser.add_argument('--run_typology','-rt', default=False, action='store_true', help="boolean for typology prediction task")
parser.add_argument('--run_phylogenetic','-rp', default=False, action='store_true', help="boolean for phylogenetic tree reconstruction task")
parser.add_argument('--run_selection','-rs', default=False, action='store_true', help="boolean for language selection task")
args = parser.parse_args()

assert os.path.isfile(args.langvec), f"{args.langvec} does not exist"
assert any([args.run_typology, args.run_phylogenetic, args.run_selection]), "No task has been set to run"

langvec = load_langvec(args.langvec)

if args.run_typology:
    walsinfo, train, dev, test_blinded, test_gold = load_data_tp(args.data_typology, args.data_wals)
    model = train_tp(langvec, train, dev)
    test_infered =infer_tp(model, langvec, test_blinded)
    score_tp(test_infered, test_gold)

if args.run_phylogenetic:
    # TODO
    pass

if args.run_selection:
    # TODO
    pass


