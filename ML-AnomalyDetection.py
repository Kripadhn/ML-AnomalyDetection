import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Load the data
data = pd.read_csv("fraud_data.csv")

# Train the model
clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(data)

# Predict the anomalies
predictions = clf.predict(data)
anomalies = data[predictions == -1]
print("Anomalies:", anomalies)
