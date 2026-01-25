import numpy as np
import os
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "data/landmarks"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS = list(string.ascii_uppercase)  # A-Z

# -----------------------------
# Load dataset
# -----------------------------
X = []
y = []

print("📂 Loading data...")

for idx, label in enumerate(LABELS):
    path = os.path.join(DATA_DIR, f"{label}.npy")
    if not os.path.exists(path):
        continue

    data = np.load(path)

    if data.ndim != 2 or data.shape[1] != 63:
        print(f"⚠️ Skipping {label} (bad shape)")
        continue

    X.extend(data)
    y.extend([idx] * len(data))
    print(f"✅ {label}: {len(data)} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n📊 Dataset summary")
print("Total samples:", X.shape[0])
print("Feature size :", X.shape[1])
print("Classes      :", len(set(y)))

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train SVM
# -----------------------------
print("\n🚀 Training SVM model...")
model = SVC(
    kernel="rbf",
    probability=True,
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Validation Accuracy: {accuracy * 100:.2f}%")
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))

# -----------------------------
# Save model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "landmark_svm_model.pkl")

if os.path.exists(model_path):
    os.remove(model_path)

joblib.dump(model, model_path)

print(f"\n✅ Model saved at: {model_path}")