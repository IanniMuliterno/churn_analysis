
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


df = pd.read_csv('ecom-user-churn-data.csv')
df.drop(['int_cat15_n'], axis= 'columns')

X = df.drop('target_class', axis=1)
y = df['target_class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler, f)

# dealing with imbalance

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# Create a dictionary for class weights
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Create sample weights for each instance
sample_weights = np.array([class_weight_dict[class_label] for class_label in y_train])

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Save the model to run in FAST API
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)