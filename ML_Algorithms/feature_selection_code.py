import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

# the dataset
cleaned_file_path = 'Cod/Data_processed_cleaned.xlsx'
df_cleaned = pd.read_excel(cleaned_file_path)

# Separate features (X) and target variable (y)
X = df_cleaned.drop(columns=['GrainYield'], errors='ignore')  # Assuming 'GrainYield' is the target variable
y = df_cleaned['GrainYield'] if 'GrainYield' in df_cleaned.columns else None


y_encoded = LabelEncoder().fit_transform(y)


if 'Longitude' in X.columns:
    X['Longitude'] = pd.to_numeric(X['Longitude'], errors='coerce')

# Mutual Information
mutual_info = mutual_info_classif(X.fillna(0), y_encoded, discrete_features='auto')
mutual_info_ranking = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

# Random Forest Feature Importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X.fillna(0), y_encoded)
rf_feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Lasso Regression for Feature Selection methods
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X.fillna(0), y_encoded)
lasso_model = SelectFromModel(lasso, prefit=True)
lasso_selected_features = X.columns[lasso_model.get_support()]

lasso_importance = pd.Series(lasso.coef_, index=X.columns).sort_values(ascending=False)

# Results
ranking_summary = pd.DataFrame({
    "Mutual_Info_Rank": mutual_info_ranking.index[:10],
    "Mutual_Info_Score": mutual_info_ranking.values[:10],
    "RF_Importance_Rank": rf_feature_importance.index[:10],
    "RF_Importance_Score": rf_feature_importance.values[:10],
    "Lasso_Importance_Rank": lasso_importance.index[:10],
    "Lasso_Importance_Score": lasso_importance.values[:10]
}).reset_index(drop=True)

# Save the results
output_path = 'Feature_Ranking_Summary_.xlsx'
ranking_summary.to_excel(output_path, index=False)
print(f"Feature ranking summary saved to {output_path}")
