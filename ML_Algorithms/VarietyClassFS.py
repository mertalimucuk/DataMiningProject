from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# Veri setini yükle
data = pd.read_excel('Cod/Data_processed_cleaned.xlsx')

variety_classes = ['VarietyClass_LDV', 'VarietyClass_SDV']
features = data.drop(columns=['GrainYield'] + variety_classes)  # VarietyClass sütunlarını kaldır
# Analysis for each VarietyClass
for variety in variety_classes:
    print(f"--- {variety} için Etkileyen Faktörler ---")
    y_variety = (data[variety] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, y_variety, test_size=0.2, random_state=42)
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Print most 10
    print(feature_importances.head(10))
    print("\n")