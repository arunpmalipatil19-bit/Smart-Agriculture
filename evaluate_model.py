import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "Crop_recommendation.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

data = pd.read_csv(DATA_FILE)

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


print("\nDataset:", len(data), "samples")
print("Features:", features)
print("Classes:", y.nunique())


# ============================================================
# 3. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# 4. 5-FOLD CROSS VALIDATION
# ============================================================

print("\nRunning 5-fold cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("\n===================================")
print("CROSS-VALIDATION RESULTS")
print("===================================")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(
    f"\nMean accuracy: {scores.mean() * 100:.2f}%"
)

print(
    f"Standard deviation: {scores.std() * 100:.2f}%"
)


# ============================================================
# 5. TRAIN MODEL FOR FEATURE IMPORTANCE
# ============================================================

model.fit(X, y)


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

for _, row in importance.iterrows():

    print(
        f"{row['Feature']:15s} "
        f"{row['Importance'] * 100:.2f}%"
    )


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

predictions = model.predict(X)

matrix = confusion_matrix(
    y,
    predictions,
    labels=model.classes_
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print("Rows/columns follow this crop order:")

print(list(model.classes_))

print("\nMatrix:")

print(matrix)


print("\n===================================")
print("VALIDATION COMPLETE")
print("===================================")