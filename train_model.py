import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "Crop_recommendation.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "crop_model.pkl"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading crop recommendation dataset...")

data = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))


# ============================================================
# 3. DISPLAY DATASET INFORMATION
# ============================================================

print("\nColumns:")
print(list(data.columns))

print("\nCrop classes:")
print(data["label"].unique())

print("\nNumber of crop classes:",
      data["label"].nunique())


# ============================================================
# 4. INPUT FEATURES AND TARGET
# ============================================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = data[features]
y = data["label"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ============================================================
# 7. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_FILE)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print("Saved at:")
print(MODEL_FILE)