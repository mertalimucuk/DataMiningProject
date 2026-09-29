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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
#
dataset_path = 'Cod/Data_processed_cleaned.xlsx'
dataframe = pd.read_excel(dataset_path)
# SoilType classes
soiltype_columns = ['SoilType_Heavy', 'SoilType_Low', 'SoilType_Medium']
# Models
evaluation_models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradBoost": GradientBoostingClassifier(),
    "HistGradBoost": HistGradientBoostingClassifier(),
    "SVC": SVC(probability=True),
    "Bayes": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "SGD": SGDClassifier(loss="log_loss")
}
# Cross-validation setup
cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# results
overall_results = {}
for soiltype_col in soiltype_columns:
    print(f"\n### Analiz yapılıyor: {soiltype_col} ###")
    # Hedef değişken ve veriyi filtreleme
    filtered_data = dataframe[dataframe[soiltype_col] == 1]
    target = filtered_data['GrainYield']
    features = filtered_data.drop(columns=['GrainYield'] + soiltype_columns)
    # SMOTE
    if len(target.unique()) > 1 and min(target.value_counts()) > 1:
        smote = SMOTE(random_state=42)
        features_resampled, target_resampled = smote.fit_resample(features, target)
    else:
        print(f"Yetersiz örnek sayısı nedeniyle SMOTE uygulanmadı: {soiltype_col}")
        features_resampled, target_resampled = features, target
    #
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_resampled)
    model_results = {}
    plt.figure(figsize=(10, 8))  # ROC eğrileri için bir figür başlat
    plt.title(f"ROC Eğrileri: {soiltype_col}")
    for model_name, model in evaluation_models.items():
        print(f"  Model: {model_name}")
        # Cross-validation
        predictions = cross_val_predict(model, scaled_features, target_resampled, cv=cross_validator, method="predict")
        # Performans
        accuracy = accuracy_score(target_resampled, predictions)
        precision = precision_score(target_resampled, predictions, average="weighted", zero_division=0)
        recall = recall_score(target_resampled, predictions, average="weighted", zero_division=0)
        f1 = f1_score(target_resampled, predictions, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(target_resampled, predictions)
        # ROC AUC calculation
        try:
            probabilities = cross_val_predict(model, scaled_features, target_resampled, cv=cross_validator, method="predict_proba")
            roc_auc = roc_auc_score(pd.get_dummies(target_resampled), probabilities, multi_class="ovr", average="weighted")
            # ROC eğrisi için
            fpr, tpr, _ = roc_curve(pd.get_dummies(target_resampled).values.ravel(), probabilities.ravel())
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
    # ROC
    plt.plot([0, 1], [0, 1], 'k--', label="Rastgele Tahmin (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{soiltype_col}.png")  # Her sınıf için ROC eğrisi kaydediliyor
    print(f"ROC eğrisi kaydedildi: roc_curve_{soiltype_col}.png")
    plt.show()
    # DataFrame
    overall_results[soiltype_col] = pd.DataFrame(model_results).T

for soiltype, results_df in overall_results.items():
    print(f"\n### Performans Sonuçları: {soiltype} ###")
    print(results_df)
    results_df.to_csv(f"results_{soiltype}.csv", index=False)
    print(f"Sonuçlar {soiltype} için 'results_{soiltype}.csv' dosyasına kaydedildi.")