import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/Datasheet.csv')
X = df.drop(columns=['team', 'FGA', 'MOV', 'NRtg', '3PAr', 'rounds'])
y = df['rounds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
num_features = ('MP', 'FG%', '3PA', '3P%', '2PA', '2P%', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Age', 'W', 'L', 
	'SOS', 'SRS', 'ORtg', 'DRtg', 'Pace', 'FTr', 'TS%', 'Offense_eFG%', 'Offense_TOV%', 'Offense_ORB%', 'Offense_FT/FGA', 'Defense_eFG%', 'Defense_TOV%', 'Defense_ORB%', 'Defense_FT/FGA')
preprocessor = ColumnTransformer([('num', StandardScaler(), num_features)], remainder='passthrough')
#single model testing:
'''
clf = Pipeline([
		('preprocessor', preprocessor),
		('model', LogisticRegression(random_state=42))
		])
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print('Training Set:')
print('Accurancy: ', np.round(clf.score(X_train, y_train), 4))
print('Precision: ', np.round(precision_score(y_train, y_pred_train, average='weighted'), 4))
print('Recall:', np.round(recall_score(y_train, y_pred_train, average='weighted'), 4))
print('F1:', np.round(f1_score(y_train, y_pred_train, average='weighted'), 4))
print('Test Set:')
print('Accurancy: ', np.round(clf.score(X_test, y_test), 4))
print('Precision: ', np.round(precision_score(y_test, y_pred_test, average='weighted'), 4))
print('Recall:', np.round(recall_score(y_test, y_pred_test, average='weighted'), 4))
print('F1:', np.round(f1_score(y_test, y_pred_test, average='weighted'), 4))
'''
models = {
	'Logistic Regression': LogisticRegression(random_state=42, max_iter=5000),
	'Decision Tree': DecisionTreeClassifier(random_state=42),
	'SVC': SVC(gamma='auto'),
	'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
	'Random Forest': RandomForestClassifier(random_state=42)
}
results = {}
for name, model in models.items():
	clf = Pipeline([
		('preprocessor', preprocessor),
		(name, model)
		])
	cross_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
	start = time.time()
	clf.fit(X_train, y_train)
	end = time.time()
	y_pred_train = clf.predict(X_train)
	results[name]={}
	results[name]['Training Accurancy'] = np.round(clf.score(X_train, y_train), 4)
	results[name]['Training Precision'] = np.round(precision_score(y_train, y_pred_train, average='weighted'), 4)
	results[name]['Training Recall'] = np.round(recall_score(y_train, y_pred_train, average='weighted'), 4)
	results[name]['Training F1'] = np.round(f1_score(y_train, y_pred_train, average='weighted'), 4)
	results[name]['Cross Validation Mean'] = np.round(cross_scores.mean(), 4)
	results[name]['Cross Validation STD'] = np.round(cross_scores.std(), 4)	
	y_pred_test = clf.predict(X_test)
	results[name]['Test Accurancy'] = np.round(clf.score(X_test, y_test), 4)
	results[name]['Test Precision'] = np.round(precision_score(y_test, y_pred_test, average='weighted'), 4)
	results[name]['Test Recall'] = np.round(recall_score(y_test, y_pred_test, average='weighted'), 4)
	results[name]['Test F1'] = np.round(f1_score(y_test, y_pred_test, average='weighted'), 4)
	results[name]['Training Time'] = end - start
	print(name)
	print(classification_report(y_test, y_pred_test))
	con_mat = confusion_matrix(y_test, y_pred_test, labels=[-1, 0, 1, 2, 3, 4])
	disp = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=clf.classes_)
	disp.plot()
	plt.title(name)
	plt.show()
results_df = pd.DataFrame.from_dict(results).T
print(results_df)