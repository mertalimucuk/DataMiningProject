import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

dataset_path = 'Cod/Data_processed_cleaned.xlsx'
dataframe = pd.read_excel(dataset_path)
# LandType
landtype_columns = ['LandType_Lowland', 'LandType_MediumLand', 'LandType_Upland']

evaluation_models = {
    "LogReg": LogisticRegression(max_iter=1000, C=1.0),
    "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_split=4, min_samples_leaf=2),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=4),
    "GradBoost": GradientBoostingClassifier(learning_rate=0.05, n_estimators=250, max_depth=8),
    "HistGradBoost": HistGradientBoostingClassifier(max_iter=200),
    "SVC": SVC(probability=True, C=2.0, kernel="rbf"),
    "Bayes": GaussianNB(var_smoothing=1e-8),
    "kNN": KNeighborsClassifier(n_neighbors=8, weights="distance"),
    "AdaBoost": AdaBoostClassifier(learning_rate=0.03, n_estimators=200),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=20),
    "MLP": MLPClassifier(max_iter=1500, hidden_layer_sizes=(50, 50), activation="relu"),
    "SGD": SGDClassifier(loss="log_loss", alpha=0.00001)
}
# Cross-validation setup
cross_validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

overall_results = {}
for landtype_col in landtype_columns:
    print(f"\n### Analiz yapılıyor: {landtype_col} ###")

    filtered_data = dataframe[dataframe[landtype_col] == 1]
    target = filtered_data['GrainYield']
    features = filtered_data.drop(columns=['GrainYield'] + landtype_columns)

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    model_results = {}
    plt.figure(figsize=(10, 8))  # ROC eğrileri için bir figür başlat
    plt.title(f"ROC Eğrileri: {landtype_col}")
    for model_name, model in evaluation_models.items():
        print(f"  Model: {model_name}")
        # Cross-validation
        predictions = cross_val_predict(model, scaled_features, target, cv=cross_validator, method="predict")

        accuracy = accuracy_score(target, predictions)
        precision = precision_score(target, predictions, average="weighted", zero_division=0)
        recall = recall_score(target, predictions, average="weighted", zero_division=0)
        f1 = f1_score(target, predictions, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(target, predictions)
        # ROC AUC calculating
        try:
            probabilities = cross_val_predict(model, scaled_features, target, cv=cross_validator, method="predict_proba")
            roc_auc = roc_auc_score(pd.get_dummies(target), probabilities, multi_class="ovr", average="weighted")
            # ROC curve
            fpr, tpr, _ = roc_curve(pd.get_dummies(target).values.ravel(), probabilities.ravel())
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        except:
            roc_auc = None
        # Save the results
        model_results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
            "AUC": roc_auc
        }
    # ROC curve drawing
    plt.plot([0, 1], [0, 1], 'k--', label="Rastgele Tahmin (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{landtype_col}.png")  # Her sınıf için ROC eğrisi kaydediliyor
    print(f"ROC eğrisi kaydedildi: roc_curve_{landtype_col}.png")
    plt.show()

    overall_results[landtype_col] = pd.DataFrame(model_results).T
# Printing all results
for landtype, results_df in overall_results.items():
    print(f"\n### Performans Sonuçları: {landtype} ###")
    print(results_df)
    results_df.to_csv(f"results_{landtype}.csv", index=False)
    print(f"Sonuçlar {landtype} için 'results_{landtype}.csv' dosyasına kaydedildi.")