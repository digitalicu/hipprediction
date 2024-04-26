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

## Experiment

Attention, certain calculations are sensitive to CPU usage and may require significant processing time.

### Test clinical judgement-based scales
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

### Run GA for search of optimal solution, example
```
python ga.py -a r -g 1000 -o m90 -f g1000_m90
```