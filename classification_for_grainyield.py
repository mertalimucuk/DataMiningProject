import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef,
    accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Function to plot ROC curves
def plot_roc_curves(models, X, y, n_classes, skf):
    plt.figure(figsize=(15, 10))
    for name, model in models.items():
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Binarize the target
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))

            # Fit the model
            pipeline = make_pipeline(StandardScaler(), model)
            pipeline.fit(X_train, y_train)

            # Predict
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)
            else:
                continue  # Skip models that don't support probabilities

            # Compute ROC curve and AUC for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

        if tprs:
            # mean
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            roc_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        else:
            print(f"No valid ROC curves generated for model: {name}")

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve for Each Model', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()

# Load dataset
data = pd.read_excel("Data_processed_cleaned.xlsx")

#
label_encoder = LabelEncoder()
data['GrainYield'] = label_encoder.fit_transform(data['GrainYield'])

# Separate features and target
X = data.drop(columns=['GrainYield'])
y = data['GrainYield']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE for balancing
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)
print("Original class distribution:", Counter(y))
print("Resampled class distribution:", Counter(y_resampled))

#  K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='ovr'),
    "kNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=10, random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), random_state=42),
    "SGD": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
}

# results
results = {name: {'Accuracy': [], 'AUC': [], 'F1': [], 'Precision': [], 'Recall': [], 'MCC': []} for name in models.keys()}

# cross-validation for each model
for name, model in models.items():
    for train_idx, test_idx in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
        y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]


        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)

        if y_pred_proba is not None:
            auc_score = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average="weighted", multi_class="ovr")
        else:
            auc_score = None

        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        # Append metrics
        results[name]['Accuracy'].append(accuracy)
        results[name]['AUC'].append(auc_score if auc_score is not None else 0)
        results[name]['F1'].append(f1)
        results[name]['Precision'].append(precision)
        results[name]['Recall'].append(recall)
        results[name]['MCC'].append(mcc)


final_results = {name: {metric: np.mean(scores) for metric, scores in metrics.items()} for name, metrics in results.items()}

# final results
print("Model Evaluation Metrics:")
for model_name, metrics in final_results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# best algorithm accuracy
best_model_name = max(final_results, key=lambda x: final_results[x]['Accuracy'])
print(f"Best Model: {best_model_name}")

# Confusion matrix for best model
best_model = models[best_model_name]
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_resampled))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title(f"Confusion Matrix for {best_model_name}")
plt.show()

# ROC plotting function
n_classes = len(np.unique(y_resampled))
plot_roc_curves(models, X_resampled, y_resampled, n_classes, skf)
