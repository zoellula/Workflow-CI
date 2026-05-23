import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train_project_model():
    # Menggunakan folder preprocessing lokal di dalam MLProject
    DATA_DIR = "studentperformance_preprocessing"
    
    # Memuat data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    
    # Melatih model sederhana untuk kebutuhan CI
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Mencatat model ke MLflow
    mlflow.sklearn.log_model(model, artifact_path="model_output")
    print("✅ Model berhasil dilatih di dalam workflow CI!")

if __name__ == "__main__":
    train_project_model()