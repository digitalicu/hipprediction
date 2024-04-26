# Enhancing Criterion Selection for Prediction Scale Construction via Genetic Algorithm: An Exemplification in Mortality Prediction for Hip Fracture

## Dependencies

[Python 3](https://www.python.org/downloads/release/python-3115/)

[PyGAD](https://pygad.readthedocs.io/en/latest/)

[Pandas](https://pandas.pydata.org/)

[SciPy](https://scipy.org/)

[scikit-learn](https://scikit-learn.org/stable/index.html)


## Data

**full.csv** - a fully anonymized dataset of patients admitted to the emergency department with hip fractures between March 2010 and April 2020 at Shamir Medical Center, Beer Yakov, Israel.

### Columns:
**m30,m90,m180,m365** - mortality for 30, 90, 180, 365 days

**procedure** - if the patient underwent surgery

**gender** - female/male

**facility** - if the patient came from a geriatric facility

**age** - complete age in years upon admission day

**total_dis** - quantity of comorbidities

**other** - additional columns include laboratory blood test findings upon admission and comorbidities, categorized by ICD-9 topics

## Modules

### ga.py

| Parameter | Description |
| -- | -- |
| -g / --generations | Number of generations (Default: 10) |
| -t / --threads | Number of threads for PyGAD lib (Default: 8) |
| -o / --outcome | Outcome m30/m90/m180/m365 (Default: m30) |
| -f / --folder | Folder to save result (Default: "default") |
| -p / --procedure | Activates column procedure as training feature (Default: False) |
| -d / --data | Data *.csv (Default: full.csv) |
| -a / --algorithm | Algorithm to use 'l', 'r', 'knn' or 'rf' |

### hand_selected_features_test.py

| Parameter | Description |
| -- | -- |
| -o / --outcome | Outcome m30/m90/m180/m365 (Default: m30) |
| -f / --features | File with JSON array of selected features (Default: "features/osteoporosis_35_561.json") |
| -p / --procedure | Activates column procedure as training feature (Default: False) |
| -d / --data | Data *.csv (Default: full.csv) |
| -a / --all_features | If active all possible features are used as scale (Default: False) |

### rfe_selected_features_test.py

| Parameter | Description |
| -- | -- |
| -o / --outcome | Outcome m30/m90/m180/m365 (Default: m30) |
| -p / --procedure | Activates column procedure as training feature (Default: False) |
| -d / --data | Data *.csv (Default: full.csv) |
| -fts / --features_to_select | Number of features for RFE to select (Default: 10) |

## Experiment

Attention, certain calculations are sensitive to CPU usage and may require significant processing time.

### Test clinical judgment-based scales
```
python hand_selected_features_test.py -o m30 -f features/gerontology_and_geriatrics_105120.json
python hand_selected_features_test.py -o m90 -f features/gerontology_and_geriatrics_105120.json
python hand_selected_features_test.py -o m180 -f features/gerontology_and_geriatrics_105120.json

python hand_selected_features_test.py -o m30 -f features/osteoporosis_35_561.json
python hand_selected_features_test.py -o m90 -f features/osteoporosis_35_561.json
python hand_selected_features_test.py -o m180 -f features/osteoporosis_35_561.json
```

### All possible data scales test
```
python hand_selected_features_test.py -o m30 -a
python hand_selected_features_test.py -o m90 -a
python hand_selected_features_test.py -o m180 -a
```

### Example of the test of GA prepared scale
```
python hand_selected_features_test.py -o m30 -f features/l_m30.json
```

### Example of RFE for special features count
```
python rfe_selected_features_test.py -o m90 -fts 24
```

### Run GA for a search of the optimal solution, example:
```
python ga.py -a r -g 1000 -o m90 -f g1000_m90
```

## Contact to author

Dr. Andrei Mazur <agrshkv@gmail.com>, [Shamir Medical Center](https://www.shamir.org/en/about/)
