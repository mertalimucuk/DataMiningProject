from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# Load data set
data = pd.read_excel('Cod/Data_processed_cleaned.xlsx')

sowing_schedules = ['SowingSchedule_T1', 'SowingSchedule_T2', 'SowingSchedule_T3',
                    'SowingSchedule_T4', 'SowingSchedule_T5']
features = data.drop(columns=['GrainYield'] + sowing_schedules)
# Analysis for each SowingSchedule
for schedule in sowing_schedules:
    print(f"--- {schedule} için Etkileyen Faktörler ---")
    y_schedule = (data[schedule] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, y_schedule, test_size=0.2, random_state=42)
    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # Feature Importance
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Print mos 10
    print(feature_importances.head(10))
    print("\n")