import os
import cv2
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

TARGET_FRUIT = 'Tomato' 

DATA_DIR = 'dataset/train'
IMG_SIZE = 64

def load_data(data_dir):
    data = []
    labels = []
    class_names = []
    
    print(f"Scanning dataset at '{data_dir}' for '{TARGET_FRUIT}'...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return np.array(data), np.array(labels), class_names

    for root, dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(root)
        
        if TARGET_FRUIT.lower() not in folder_name.lower():
            continue
        if root == data_dir:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                if folder_name not in class_names:
                    class_names.append(folder_name)
                
                label_index = class_names.index(folder_name)
                
                try:
                    img_array = cv2.imread(img_path)
                    if img_array is not None:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        img_flat = img_resized.flatten()
                        data.append(img_flat)
                        labels.append(label_index)
                except Exception:
                    pass
    
    return np.array(data), np.array(labels), class_names

X, y, class_names = load_data(DATA_DIR)

if len(X) == 0:
    print("No images found! Check dataset or target fruit name.")
    exit()
else:
    print(f"Loaded {len(X)} images.")
    print(f"Classes: {class_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "lr": LogisticRegression(max_iter=1000),
    "svm_rbf": SVC(kernel='rbf', probability=True),
    "svm_linear": SVC(kernel='linear', probability=True),
    "dt": DecisionTreeClassifier(random_state=42),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "nb": GaussianNB()
}

r = {}

print("\nStarting Training...\n")
print(f"{'Model':<20} | {'Accuracy':<10}")
print("-" * 35)

for name, model in models.items():
    pipeline = Pipeline([
        ("sc", StandardScaler()),
        ("model", model)
    ])
    
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        r[name] = {
            "model": pipeline,
            "accuracy": score
        }
        
        print(f"{name:<20} | {score:.4f}")
        
    except Exception as e:
        print(f"{name:<20} | Failed ({e})")

best_name = max(r, key=lambda x: r[x]["accuracy"])
best_pipeline = r[best_name]["model"]
best_score = r[best_name]["accuracy"]

print("-" * 35)
print(f"Best Model: {best_name} (Accuracy: {best_score:.4f})")

model_filename = f"best_{TARGET_FRUIT.lower()}_model.pkl"
joblib.dump(best_pipeline, model_filename)
print(f"Saved model to '{model_filename}'")

class_file = f"{TARGET_FRUIT.lower()}_class_names.json"
with open(class_file, 'w') as f:
    json.dump(class_names, f)
print(f"Saved class names to '{class_file}'")
