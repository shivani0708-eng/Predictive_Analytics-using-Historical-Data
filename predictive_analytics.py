import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Read dataset
df = pd.read_csv("historical_data.csv")

print(df.head())
print(df.info())

# Features
X = df[['Year']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future_years = pd.DataFrame({
    'Year': [2025, 2026, 2027, 2028, 2029]
})

predictions = model.predict(future_years)

future_years['PredictedSales'] = predictions

print(future_years)

# Save predictions
future_years.to_csv("predicted_data.csv", index=False)

# Graph
plt.figure(figsize=(8,6))

# Historical Data
plt.scatter(
    df['Year'],
    df['Sales'],
    color='blue',
    label='Historical Data'
)

# Regression Line
plt.plot(
    df['Year'],
    model.predict(X),
    color='green',
    label='Regression Line'
)

# Future Prediction
plt.scatter(
    future_years['Year'],
    future_years['PredictedSales'],
    color='red',
    marker='X',
    s=120,
    label='Prediction'
)

plt.xlabel("Year")
plt.ylabel("Sales")
plt.title("Predictive Analytics Using Historical Data")
plt.legend()

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Save graph
plt.savefig("prediction_graph.png", dpi=300, bbox_inches="tight")

# Show graph
plt.show()