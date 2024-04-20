from data import DataManager
import argparse
import json
from ml import mlp_classifier

parser = argparse.ArgumentParser()
parser.add_argument("-o", 
                    "--outcome",
                    help="Outcome to check: 'm30', 'm90', 'm180', 'm365'", 
                    default="m30",
                    type=str)
parser.add_argument("-l", 
                    "--layers",
                    help="File with json array of hidden layers sizes for MLP", 
                    default="layers/m30.json",
                    type=str)
parser.add_argument("-p", 
                    "--procedure",
                    help="To include procedure as parameter", 
                    action=argparse.BooleanOptionalAction,
                    type=bool)
parser.add_argument("-d", 
                    "--data",
                    help="Data *.csv", 
                    default="full.csv",
                    type=str)
args = parser.parse_args()

dm = DataManager(args.data)
dm.set_feature(args.outcome)

if args.procedure:
    dm.switch_procedure()

layers = []
with open(args.layers, "r") as layers_file:
    layers = json.loads(layers_file.read())

print("Source: %s" % args.layers)
print("MLP AUC: %.6f" % mlp_classifier(dm, hidden_layers=layers))