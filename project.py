import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# LOAD DATASET
# ==========================

file_path = r"C:\Users\nb493\OneDrive\Documents\python programs\dataset.csv"
aqi_data = pd.read_csv(file_path)

print("Dataset Information:")
print(aqi_data.info())

print("\nMissing Values:")
print(aqi_data.isnull().sum())

# Remove missing values
aqi_data = aqi_data.dropna()

# ==========================
# EXPLORATORY DATA ANALYSIS
# ==========================

plt.figure(figsize=(10,6))
sns.histplot(aqi_data['AQI Value'], kde=True, bins=30)
plt.title("Distribution of AQI Values")
plt.show()

# Top 10 Highest AQI Countries

plt.figure(figsize=(12,6))
sns.barplot(
    data=aqi_data.nlargest(10,'AQI Value'),
    x='Country',
    y='AQI Value'
)

plt.xticks(rotation=45)
plt.title("Top 10 Countries with Highest AQI")
plt.show()

# ==========================
# AQI CATEGORY CREATION
# ==========================

def categorize_aqi(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


aqi_data['AQI_Category'] = aqi_data['AQI Value'].apply(categorize_aqi)

plt.figure(figsize=(10,6))

sns.countplot(
    data=aqi_data,
    x='AQI_Category',
    order=[
        'Good',
        'Moderate',
        'Unhealthy for Sensitive Groups',
        'Unhealthy',
        'Very Unhealthy',
        'Hazardous'
    ]
)

plt.xticks(rotation=20)
plt.title("AQI Category Distribution")
plt.show()

# ==========================
# FEATURE SELECTION
# ==========================

X = aqi_data[
[
'CO AQI Value',
'Ozone AQI Value',
'NO2 AQI Value',
'PM2.5 AQI Value',
'SO2'
]
]
y = aqi_data['AQI Value']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# RANDOM FOREST MODEL
# ==========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

# ==========================
# MODEL EVALUATION
# ==========================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MSE :", mse)
print("R2 Score :", r2)

# ==========================
# ACTUAL VS PREDICTED
# ==========================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()

# ==========================
# USER INPUT PREDICTION
# ==========================

lat = float(input("Enter Latitude: "))
lng = float(input("Enter Longitude: "))

new_data = pd.DataFrame({
    'lat': [lat],
    'lng': [lng]
})

predicted_aqi = model.predict(new_data)

print("\nPredicted AQI:", predicted_aqi[0])
