from data import DataManager
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse
from ml import logistic_regression, random_forest, rfe_feature_selection

parser = argparse.ArgumentParser()
parser.add_argument("-o", 
                    "--outcome",
                    help="Outcome to check: 'm30', 'm90', 'm180', 'm365'", 
                    default="m30",
                    type=str)
parser.add_argument("-d", 
                    "--data",
                    help="Data *.csv", 
                    default="full.csv",
                    type=str)
parser.add_argument("-fts", 
                    "--features_to_select",
                    help="Number of features for RFE to select", 
                    default=10,
                    type=int)
parser.add_argument("-p", 
                    "--procedure",
                    help="To include procedure as parameter", 
                    action=argparse.BooleanOptionalAction,
                    type=bool)
args = parser.parse_args()

dm = DataManager(args.data)
dm.set_feature(args.outcome)

if args.procedure:
    dm.switch_procedure()

lr_features = rfe_feature_selection(dm, LogisticRegression(solver='liblinear'), fts=args.features_to_select)
dm.set_tested_features(lr_features)
print("Selected: %s" % ", ".join(lr_features))
print("Logistic regression AUC: %.6f" % logistic_regression(dm))

rf_features = rfe_feature_selection(dm, RandomForestClassifier(n_estimators=100), fts=args.features_to_select)
dm.set_tested_features(rf_features)
print("Selected: %s" % ", ".join(rf_features))
print("Random Forest AUC:       %.6f" % random_forest(dm))