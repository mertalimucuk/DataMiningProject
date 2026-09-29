import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


dataset_path = 'Cod/Data_processed_cleaned.xlsx'
dataframe = pd.read_excel(dataset_path)

# CropEstablishment
cropestablishment_columns = ['CropEstablishment_CT', 'CropEstablishment_ZT', 'CropEstablishment_CT_line']

# Models
evaluation_models = {
    "LogReg": LogisticRegression(max_iter=1500, C=1.2),
    "DecisionTree": DecisionTreeClassifier(max_depth=15, min_samples_split=5),
    "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_split=4),
    "GradBoost": GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=8),
    "HistGradBoost": HistGradientBoostingClassifier(max_iter=200),
    "SVC": SVC(probability=True, C=2.0, kernel="rbf"),
    "Bayes": GaussianNB(var_smoothing=1e-9),
    "kNN": KNeighborsClassifier(n_neighbors=10, weights="distance"),
    "AdaBoost": AdaBoostClassifier(learning_rate=0.03, n_estimators=150),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=20),
    "MLP": MLPClassifier(max_iter=2000, hidden_layer_sizes=(100, 50), activation="relu"),
    "SGD": SGDClassifier(loss="log_loss", alpha=0.0001)
}

# Cross-validation
cross_validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


overall_results = {}

for cropestablishment_col in cropestablishment_columns:
    print(f"\n### Analiz yapılıyor: {cropestablishment_col} ###")


    filtered_data = dataframe[dataframe[cropestablishment_col] == 1]
    target = filtered_data['GrainYield']

    # Özellikleri temizleme
    features = filtered_data.drop(columns=['GrainYield'] + cropestablishment_columns)
    features = features.applymap(lambda x: str(x).strip().replace('\xa0', '') if isinstance(x, str) else x)
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1)

    # SMOTE
    if len(target.unique()) > 1 and min(target.value_counts()) > 1:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(target.value_counts()) - 1))
        features_resampled, target_resampled = smote.fit_resample(features, target)
    else:
        print(f"Yetersiz örnek sayısı nedeniyle SMOTE uygulanmadı: {cropestablishment_col}")
        features_resampled, target_resampled = features, target


    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_resampled)

    model_results = {}
    plt.figure(figsize=(10, 8))  # ROC eğrileri için bir figür başlat
    plt.title(f"ROC Eğrileri: {cropestablishment_col}")

    for model_name, model in evaluation_models.items():
        print(f"  Model: {model_name}")

        # Cross-validation
        predictions = cross_val_predict(model, scaled_features, target_resampled, cv=cross_validator, method="predict")

        # Performans metrikleri
        accuracy = accuracy_score(target_resampled, predictions)
        precision = precision_score(target_resampled, predictions, average="weighted", zero_division=0)
        recall = recall_score(target_resampled, predictions, average="weighted", zero_division=0)
        f1 = f1_score(target_resampled, predictions, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(target_resampled, predictions)

        # ROC AUC
        try:
            probabilities = cross_val_predict(model, scaled_features, target_resampled, cv=cross_validator,
                                              method="predict_proba")
            roc_auc = roc_auc_score(pd.get_dummies(target_resampled), probabilities, multi_class="ovr",
                                    average="weighted")

            # ROC eğrisi için
            fpr, tpr, _ = roc_curve(pd.get_dummies(target_resampled).values.ravel(), probabilities.ravel())
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        except:
            roc_auc = None

        # Sonuçları kaydet
        model_results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
            "AUC": roc_auc
        }

    # ROC eğrisini
    plt.plot([0, 1], [0, 1], 'k--', label="Rastgele Tahmin (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{cropestablishment_col}.png")  # Her sınıf için ROC eğrisi kaydediliyor
    print(f"ROC eğrisi kaydedildi: roc_curve_{cropestablishment_col}.png")
    plt.show()

    # Sonuçları DataFrame olarak sakla
    overall_results[cropestablishment_col] = pd.DataFrame(model_results).T

# All results
for cropestablishment, results_df in overall_results.items():
    print(f"\n### Performans Sonuçları: {cropestablishment} ###")
    print(results_df)
    results_df.to_csv(f"results_{cropestablishment}.csv", index=False)
    print(f"Sonuçlar {cropestablishment} için 'results_{cropestablishment}.csv' dosyasına kaydedildi.")

# Best model
best_algorithms = {}

for cropestablishment, results_df in overall_results.items():
    best_model_name = results_df['Accuracy'].idxmax()
    best_model_accuracy = results_df.loc[best_model_name, 'Accuracy']
    best_algorithms[cropestablishment] = (best_model_name, best_model_accuracy)
    print(f"\n{cropestablishment} için en iyi model: {best_model_name} (Accuracy: {best_model_accuracy:.4f})")

    # Confusion matrix
    best_model = evaluation_models[best_model_name]


    filtered_data = dataframe[dataframe[cropestablishment] == 1]
    target = filtered_data['GrainYield']
    features = filtered_data.drop(columns=['GrainYield'] + cropestablishment_columns)
    features = features.applymap(lambda x: str(x).strip().replace('\xa0', '') if isinstance(x, str) else x)
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.dropna(axis=1)

    # Apply SMOTE
    if len(target.unique()) > 1 and min(target.value_counts()) > 1:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(target.value_counts()) - 1))
        features_resampled, target_resampled = smote.fit_resample(features, target)
    else:
        features_resampled, target_resampled = features, target


    scaled_features = scaler.fit_transform(features_resampled)

    # Cross-validation
    predictions = cross_val_predict(best_model, scaled_features, target_resampled, cv=cross_validator)

    # Confusion matrix calculate
    cm = confusion_matrix(target_resampled, predictions)

    # Confusion matrix visulation
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(target_resampled),
                yticklabels=np.unique(target_resampled))
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f"Confusion Matrix: {cropestablishment} - {best_model_name}")
    plt.savefig(f"confusion_matrix_{cropestablishment}.png")
    print(f"Confusion matrix kaydedildi: confusion_matrix_{cropestablishment}.png")
    plt.show()
