from data import DataManager
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import argparse
from ml import lasso, ridge, random_forest, rfe_feature_selection, knn, gnb

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

print("Outcome: %s" % args.outcome)

lr_features = rfe_feature_selection(dm, LogisticRegression(solver='liblinear', penalty="l2"), fts=args.features_to_select)
dm.set_tested_features(lr_features)
print("Selected: %s" % ", ".join(lr_features))
print("LR (Ridge) AUC:          %.6f" % ridge(dm))
dm.set_tested_features(None)

l_features = rfe_feature_selection(dm, LogisticRegression(solver='liblinear', penalty="l1"), fts=args.features_to_select)
dm.set_tested_features(l_features)
print("Selected: %s" % ", ".join(l_features))
print("LR (LASSO) AUC:          %.6f" % lasso(dm))
dm.set_tested_features(None)

rf_features = rfe_feature_selection(dm, RandomForestClassifier(n_estimators=100), fts=args.features_to_select)
dm.set_tested_features(rf_features)
print("Selected: %s" % ", ".join(rf_features))
print("Random Forest AUC:       %.6f" % random_forest(dm))
dm.set_tested_features(None)

print("KNN test is unavailable")
print("Naive Bayes test is unavailable")