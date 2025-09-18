from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# Load dataset
data = pd.read_excel('Cod/Data_processed_cleaned.xlsx')

# Hedef değişken ve özellikler
soiltypes = ['SoilType_Low', 'SoilType_Medium', 'SoilType_Heavy']
features = data.drop(columns=['GrainYield'] + soiltypes)  # SoilType sütunlarını kaldır
# Analys for each SoilType
for soiltype in soiltypes:
    print(f"--- {soiltype} için Etkileyen Faktörler ---")
    y_soiltype = (data[soiltype] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, y_soiltype, test_size=0.2, random_state=42)
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Printing most 10
    print(feature_importances.head(10))
    print("\n")