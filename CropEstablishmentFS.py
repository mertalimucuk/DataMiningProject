from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# Load data set
data = pd.read_excel('Cod/Data_processed_cleaned.xlsx')


crop_establishments = ['CropEstablishment_CT', 'CropEstablishment_CT_line', 'CropEstablishment_ZT']
features = data.drop(columns=['GrainYield'] + crop_establishments)  # CropEstablishment sütunlarını kaldır
# Each CropEstablishment analysis
for establishment in crop_establishments:
    print(f"--- {establishment} için Etkileyen Faktörler ---")
    y_establishment = (data[establishment] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, y_establishment, test_size=0.2, random_state=42)
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Print most 10
    print(feature_importances.head(10))
    print("\n")