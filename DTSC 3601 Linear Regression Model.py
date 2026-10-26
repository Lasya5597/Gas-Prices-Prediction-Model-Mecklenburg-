import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv(r"C:\Users\Rachel\Downloads\DTSC 3601\charlotte_nc_gas_prices_weekly.csv")

# Make sure the dates are dates
data["Week_Ending"] = pd.to_datetime(data["Week_Ending"])

# Sort by date
data = data.sort_values("Week_Ending")

# Turn dates into numbers
data["day_number"] = (
    data["Week_Ending"] - data["Week_Ending"].min()
).dt.days

# X = what we use to make the prediction
# y = what we are trying to predict
X = data[["day_number"]]
y = data["Regular"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Print the equation
print("Linear Regression Equation:")
print(f"Gas Price = {model.intercept_:.2f} + {model.coef_[0]:.4f} × Day Number")

# Print R-squared
print(f"\nR-squared: {model.score(X, y):.2f}")

# Make predictions for the historical data
predicted = model.predict(X)

# Create future dates
future_dates = pd.date_range(
    start="2026-10-04",
    periods=52,
    freq="W-SUN"
)

future_days = (
    future_dates - data["Week_Ending"].min()
).days

# Predict future gas prices
future_predictions = model.predict(
    pd.DataFrame({"day_number": future_days})
)

# Print future predictions
print("\nFuture Gas Price Predictions:")
for date, price in zip(future_dates, future_predictions):
    print(date.date(), f"${price:.2f}")

# Graph the actual prices
plt.scatter(data["Week_Ending"], y, label="Actual Prices")

# Graph the regression line for historical data
plt.plot(data["Week_Ending"], predicted, label="Regression Line")

# Graph the future predictions
plt.plot(
    future_dates,
    future_predictions,
    label="Future Predictions"
)

plt.xlabel("Date")
plt.ylabel("Gas Price ($)")
plt.title("Charlotte Gas Prices - Linear Regression")
plt.legend()
plt.xticks(rotation=45)
plt.show()