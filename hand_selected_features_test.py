from data import DataManager
import argparse
import json
from ml import logistic_regression, random_forest

parser = argparse.ArgumentParser()
parser.add_argument("-o", 
                    "--outcome",
                    help="Outcome to check: 'm30', 'm90', 'm180', 'm365'", 
                    default="m30",
                    type=str)
parser.add_argument("-f", 
                    "--features",
                    help="File with json array of selected features", 
                    default="features/osteoporosis_35_561.json",
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

tested_features = []
with open(args.features, "r") as features_file:
    tested_features = json.loads(features_file.read())

dm.set_tested_features(tested_features)

print("Source: %s" % args.features)
print("Number of features: %s" % len(tested_features))
print("Logistic regression AUC: %.6f" % logistic_regression(dm))
print("Random Forest AUC:       %.6f" % random_forest(dm))