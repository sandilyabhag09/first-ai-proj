# Screen Time Predictor
# Predict daily_active_minutes_instagram based on user demographics and lifestyle

import pandas as pd
import numpy as np
import joblib
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('/Users/sandilyabhagavatula/Desktop/social-media-model-novibe/instagram_usage_lifestyle.csv')

print(df.head())
print(df.info())
print(df.shape)

X = df[['social_events_per_month', 'followers_count', 'relationship_status', 'sleep_hours_per_night', 'self_reported_happiness', 'age', 'gender', 'country', 'employment_status', 'education_level', 'perceived_stress_score', 'weekly_work_hours', 'hobbies_count', 'income_level']]
y = df['daily_active_minutes_instagram']

X = X.copy()
X['country'] = preprocessing.LabelEncoder().fit_transform(X['country'])
X['education_level'] = preprocessing.LabelEncoder().fit_transform(X['education_level'])
X = pd.get_dummies(X, columns=['relationship_status', 'income_level', 'gender', 'employment_status'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=104, shuffle=True)

model = ensemble.GradientBoostingRegressor(n_estimators=70, max_depth=5, random_state=104)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(metrics.r2_score(y_test, y_pred))

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]

for name, importance in zip(names, importances[indices]):
    print(name, importance)

joblib.dump(model, 'screen_time_model.joblib')
