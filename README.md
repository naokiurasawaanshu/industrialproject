import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib

# --- Data Loading and Preprocessing ---
# Load the dataset
df = pd.read_csv('data/sensor_readings.csv')

# Separate features (X) and target (y)
X = df.drop(['timestamp', 'machine_failure'], axis=1)
y = df['machine_failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler for later use in prediction
joblib.dump(scaler, "models/scaler.pkl")

# --- Model Training and Evaluation ---
# Define the models to be trained
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
}

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"--- Training {name} ---")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    # Print evaluation report
    print(f"\n{name} Results:")
    print(classification_report(y_test, preds))
    print("---" * 10)

    # Save the trained model
    joblib.dump(model, f"models/model_{name.lower().replace(' ', '_')}.pkl")

print("\nAll models have been trained and saved.")


import joblib
import pandas as pd

# --- Load Model and Scaler ---
# Load the pre-trained model and scaler
model = joblib.load("models/model_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# --- Prepare Input Data ---
# Create a DataFrame for a new data point
sample_data = pd.DataFrame([{
    "temperature": 90,
    "vibration": 0.04,
    "pressure": 31,
    "heating_level": 65
}])

# Scale the input data using the loaded scaler
sample_scaled = scaler.transform(sample_data)

# --- Make Prediction ---
# Use the model to predict the outcome
prediction = model.predict(sample_scaled)

# Print the result
if prediction[0] == 1:
    print("Prediction: Machine is likely to fail.")
else:
    print("Prediction: Machine is likely to be fine.") 
