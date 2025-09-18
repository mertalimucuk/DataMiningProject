import pandas as pd

# Load the dataset
file_path = 'Cod/Data_processed.xlsx'
data = pd.ExcelFile(file_path)

# Load the data from the first sheet
df = data.parse('05 - data for classif one hot')

# Cleaning the 'Longitude' column (example: '85.1\xa04')
df['Longitude'] = pd.to_numeric(df['Longitude'].astype(str).str.replace(r'\xa0', '', regex=True), errors='coerce')

# Fill missing values in 'Longitude' with the column mean
df['Longitude'] = df['Longitude'].fillna(df['Longitude'].mean())

# Fill missing values for feature columns only
df_cleaned = df.copy()

# Exclude the 'GrainYield' column (class attribute) from cleaning
class_column = 'GrainYield'

for column in df_cleaned.columns:
    if column != class_column:  # Skip the class attribute
        if df_cleaned[column].dtype == 'object':  # Convert non-numeric columns
            df_cleaned[column] = pd.to_numeric(df_cleaned[column].astype(str).str.replace(r'\xa0', '', regex=True), errors='coerce')
        df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())

# Save the cleaned dataset
cleaned_file_path = 'Cod/Data_processed_cleaned.xlsx'
df_cleaned.to_excel(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")
