from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# Load data set
data = pd.read_excel('Cod/Data_processed_cleaned.xlsx')

# STATE
states = ['State_Bihar', 'State_UP']
features_state = data.drop(columns=['GrainYield'] + states)  # State sütunlarını kaldır
print("### STATE ANALİZİ ###")
for state in states:
    print(f"--- {state} için Etkileyen Faktörler ---")
    y_state = (data[state] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features_state, y_state, test_size=0.2, random_state=42)
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Print most 10
    print(feature_importances.head(10))
    print("\n")
# LANDTYPE analysis
landtypes = ['LandType_Lowland', 'LandType_MediumLand', 'LandType_Upland']
features_landtype = data.drop(columns=['GrainYield'] + landtypes)
print("### LANDTYPE ANALİZİ ###")
for landtype in landtypes:
    print(f"--- {landtype} için Etkileyen Faktörler ---")
    y_landtype = (data[landtype] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features_landtype, y_landtype, test_size=0.2, random_state=42)
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Print most 10
    print(feature_importances.head(10))
    print("\n")