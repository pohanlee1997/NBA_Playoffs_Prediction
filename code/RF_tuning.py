import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/Datasheet.csv')
X = df.drop(columns=['team', 'FGA', 'MOV', 'NRtg', '3PAr', 'rounds'])
y = df['rounds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
num_features = ('MP', 'FG%', '3PA', '3P%', '2PA', '2P%', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Age', 'W', 'L', 
	'SOS', 'SRS', 'ORtg', 'DRtg', 'Pace', 'FTr', 'TS%', 'Offense_eFG%', 'Offense_TOV%', 'Offense_ORB%', 'Offense_FT/FGA', 'Defense_eFG%', 'Defense_TOV%', 'Defense_ORB%', 'Defense_FT/FGA')
preprocessor = ColumnTransformer([('num', StandardScaler(), num_features)], remainder='passthrough')

clf = Pipeline([
	('preprocessor', preprocessor),
	('model', RandomForestClassifier(random_state=42))
	])
param_grid = {
	'model__n_estimators' :[100, 200, 300],
	'model__max_depth': [None, 30, 20, 10],
	'model__class_weight': [None, 'balanced'],
	'model__max_features': [None, 'sqrt', 'log2'],
	'model__min_samples_split': [2, 4, 6, 8]
	}
grid_search = GridSearchCV(
	estimator = clf,
	param_grid = param_grid,
	scoring = 'f1_weighted',
	cv = 5,
	)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print('CV score:', np.round(grid_search.best_score_, 4))