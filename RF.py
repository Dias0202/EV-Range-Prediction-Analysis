import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Set up visualization style
plt.style.use('ggplot')  # Changed from 'seaborn' which was causing errors
sns.set_palette("viridis")

# 1. Load and prepare data
print("Loading and preparing data...")
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
df = df[["Model Year", "Make", "Model", "Electric Vehicle Type", "Electric Range"]].dropna()
df = df[(df["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)") & (df["Electric Range"] > 0)]

# 2. Identify top performing models (top 20% by range)
top_threshold = df["Electric Range"].quantile(0.8)
top_models = df[df["Electric Range"] >= top_threshold]["Model"].unique()
df["Is_Top_Model"] = df["Model"].isin(top_models)

# 3. Feature engineering
print("Creating advanced features...")
df['Make_Model'] = df['Make'].str.upper() + '_' + df['Model'].str.upper()
df['Years_Since_2010'] = df['Model Year'] - 2010

# 4. Build and train model
print("Training main model...")
X = df[['Make_Model', 'Years_Since_2010', 'Is_Top_Model']]
y = df['Electric Range']

model = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Make_Model']),
        ('num', StandardScaler(), ['Years_Since_2010']),
        ('bool', 'passthrough', ['Is_Top_Model'])
    ])),
    ('regressor', GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")


def predict_future_range(make, model, start_year=2025, end_year=2030):
    """Predict future electric range for a specific make and model"""
    make_model = f"{make.upper()}_{model.upper()}"
    years = np.array(range(start_year, end_year + 1))

    # Calculate historical trend if data exists
    if make_model in df['Make_Model'].unique():
        historical = df[df['Make_Model'] == make_model]
        trend = np.polyfit(historical['Model Year'], historical['Electric Range'], 1)
        base_pred = np.poly1d(trend)(years)
    else:
        overall_trend = np.polyfit(df['Model Year'], df['Electric Range'], 1)
        base_pred = np.poly1d(overall_trend)(years)

    # Apply manufacturer-specific improvement factors
    if make_model == 'TESLA_MODEL 3':
        improvement = np.linspace(1.0, 1.20, len(years))  # 20% improvement for Tesla
    elif make_model == 'NISSAN_LEAF':
        improvement = np.linspace(1.0, 1.10, len(years))  # 10% improvement for Nissan
    else:
        improvement = np.linspace(1.0, 1.15, len(years))  # 15% default improvement

    return pd.DataFrame({
        'Model Year': years,
        'Adjusted_Prediction': base_pred * improvement,
        'Improvement_Factor': improvement
    })


def predict_top_models_average(start_year=2025, end_year=2030):
    """Predict average range for top performing models"""
    years = np.array(range(start_year, end_year + 1))
    avg_predictions = []

    for year in years:
        synthetic_data = pd.DataFrame({
            'Make_Model': ['AVERAGE_TOP_MODELS'],
            'Years_Since_2010': [year - 2010],
            'Is_Top_Model': [True]
        })
        avg_predictions.append(model.predict(synthetic_data)[0])

    # Apply 18% technology improvement factor
    improvement = np.linspace(1.0, 1.18, len(years))
    adjusted_predictions = np.array(avg_predictions) * improvement

    return pd.DataFrame({
        'Model Year': years,
        'Average_Top_Models': avg_predictions,
        'Adjusted_Prediction': adjusted_predictions,
        'Improvement_Factor': improvement
    })


# Generate predictions
print("\nGenerating predictions...")
tesla_pred = predict_future_range("TESLA", "Model 3")
nissan_pred = predict_future_range("NISSAN", "Leaf")
top_models_avg = predict_top_models_average()

# Create visualization
print("\nCreating visualizations...")
plt.figure(figsize=(14, 8))

# Plot predictions
plt.plot(tesla_pred['Model Year'], tesla_pred['Adjusted_Prediction'],
         '.-', linewidth=2, label='Tesla Model 3', color='#ff7f0e')
plt.plot(nissan_pred['Model Year'], nissan_pred['Adjusted_Prediction'],
         '.-', linewidth=2, label='Nissan Leaf', color='#1f77b4')
plt.plot(top_models_avg['Model Year'], top_models_avg['Adjusted_Prediction'],
         '.-', linewidth=3, label='Top Models Average', color='#2ca02c', linestyle='--')

# Chart formatting
plt.title('Electric Vehicle Range Projection (2025-2030)\nComparison with Top Models Average', pad=20)
plt.xlabel('Year', labelpad=10)
plt.ylabel('Range (miles)', labelpad=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Add data table
col_labels = ['Year', 'Tesla Model 3', 'Nissan Leaf', 'Top Models Avg']
table_vals = [
    top_models_avg['Model Year'].astype(int).tolist(),
    np.round(tesla_pred['Adjusted_Prediction'], 1).tolist(),
    np.round(nissan_pred['Adjusted_Prediction'], 1).tolist(),
    np.round(top_models_avg['Adjusted_Prediction'], 1).tolist()
]

plt.table(cellText=np.array(table_vals).T,
          colLabels=col_labels,
          cellLoc='center',
          loc='bottom',
          bbox=[0, -0.5, 1, 0.3])

plt.subplots_adjust(left=0.2, bottom=0.3)

# Show the plot
plt.show()

# Comparative analysis
print("\nComparative Analysis:")
comparison_df = pd.DataFrame({
    'Year': top_models_avg['Model Year'],
    'Tesla_Model_3': tesla_pred['Adjusted_Prediction'],
    'Nissan_Leaf': nissan_pred['Adjusted_Prediction'],
    'Top_Models_Avg': top_models_avg['Adjusted_Prediction']
})

comparison_df['Tesla_vs_Avg'] = comparison_df['Tesla_Model_3'] - comparison_df['Top_Models_Avg']
comparison_df['Nissan_vs_Avg'] = comparison_df['Nissan_Leaf'] - comparison_df['Top_Models_Avg']

print(comparison_df.round(1))