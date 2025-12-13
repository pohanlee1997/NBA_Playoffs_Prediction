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

df = pd.read_csv('../data/Datasheet.csv')
X = df.drop(columns=['team', 'FGA', 'MOV', 'NRtg', '3PAr', 'rounds'])
y = df['rounds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
num_features = ('MP', 'FG%', '3PA', '3P%', '2PA', '2P%', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Age', 'W', 'L', 
	'SOS', 'SRS', 'ORtg', 'DRtg', 'Pace', 'FTr', 'TS%', 'Offense_eFG%', 'Offense_TOV%', 'Offense_ORB%', 'Offense_FT/FGA', 'Defense_eFG%', 'Defense_TOV%', 'Defense_ORB%', 'Defense_FT/FGA')
preprocessor = ColumnTransformer([('num', StandardScaler(), num_features)], remainder='passthrough')

clf = Pipeline([
	('preprocessor', preprocessor),
	('model', LogisticRegression(random_state=42, max_iter=5000))
	])

param_grid =[
	{
		'model__penalty': ['elasticnet'],
		'model__solver': ['saga'],
		'model__class_weight': [None, 'balanced'],
		'model__C': [0.01, 0.1, 1.0, 10],
		'model__l1_ratio': [0.2, 0.4, 0.6, 0.8],
		'model__tol':[0.0001, 0.001, 0.01]
	},

	{
		'model__penalty': ['l1'],
		'model__solver': ['liblinear', 'saga'],
		'model__class_weight': [None, 'balanced'],
		'model__C': [0.01, 0.1, 1.0, 10],
		'model__tol':[0.0001, 0.001, 0.01]
	},

	{
		#ovr
		'model__penalty': ['l2'],
		'model__solver': ['liblinear'],
		'model__class_weight': [None, 'balanced'],
		'model__C': [0.01, 0.1, 1.0, 10],
		'model__tol':[0.0001, 0.001, 0.01]
	},

	{
		'model__penalty': ['l2'],
		'model__solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
		'model__class_weight': [None, 'balanced'],
		'model__C': [0.01, 0.1, 1.0, 10],
		'model__tol':[0.0001, 0.001, 0.01]
	},

	{
		'model__penalty': [None],
		'model__solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
		'model__class_weight': [None, 'balanced'],
		'model__C': [1.0],
		'model__tol':[0.0001, 0.001, 0.01]
	}
]

grid_search = GridSearchCV(
	estimator = clf,
	param_grid = param_grid,
	scoring = 'f1_weighted',
	cv = 5,
	)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print('CV score:', np.round(grid_search.best_score_, 4))