### Environment Setup

This project was developed using **Python 3.13.7** and requires the following libraries.
* pandas 
* numpy 
* scikit-learn 
* imbalanced-learn

### How to Run the Project
This project consists of 3 main stages: Data Collection, Data Exploration, and Machine Learning Experiments.
1. Data Collection(Optional)
   Note: All datasets are provided in the data/ directory. You may directly use them.

   To colloect the data from online sources
```bat
python3 get_label.py
python3 awards.py
python3 get_features_teams_stats.py 
```
   And then combine them into a single dataset
```
Python3 DataSheet.py
```
2. Data Exploration
To expolore the data distribution and features correlations
```
python3 eda.py
```
4. Machine Learning Experiments
Initial run:
```
python3 training_baseline.py
```
Logistic Regression:
Hyperparameter tuning and sampling experiments:
```
python3 LR_tuning.py
python3 LR_sampling.py
python3 LR_sampled_tuning.py
```
Final Logistic Regression results:
```
python3 LR_tuned.py
python3 LR_samplid.py
python3 LR_sampled_tuned.py
```
Random Forest:
Hyperparameter tuning and sampling experiments:
```
python3 RF_tuning.py
python3 RF_sampling.py
python3 RF_sampled_tuning.py 
```
Final Random Forest results:
```
python3 RF_tuned.py
python3 RF_sampled.py
python3 RF_sampled_tuned.py
```
