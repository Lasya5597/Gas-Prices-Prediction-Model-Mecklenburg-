# Charlotte Gas Price Predictor

A predictive analytics tool that forecasts average regular gas prices for the Charlotte, NC metro area. It pulls current price data through the GitHub API, stores it in Amazon S3, trains a linear regression model on weekly historical prices, and produces future price predictions (including a CSV for ArcGIS mapping).

Our long-term goal is a **cost savings by gas station distance index** that helps users decide whether driving farther to a cheaper station is actually worth the extra fuel.

> **Course:** DTSC 3601: Data Science Studio 3, Predictive Analysis & Their Impacts
> **Group:** Lasya Maddula, Rachel Roberts, Minna Abunura, Ruhan Rahman 

## Table of Contents

- [Installation](#installation)
- [Get Started](#get-started)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Planned Classes](#planned-classes)
- [Examples](#examples)
- [Contact](#contact)
- [License](#license)

## Installation

### Required libraries

- Python 3.9 or newer
- [pandas](https://pandas.pydata.org/) (data handling)
- [scikit-learn](https://scikit-learn.org/) (`LinearRegression` model)
- [matplotlib](https://matplotlib.org/) (plots)
- [requests](https://requests.readthedocs.io/) (GitHub API calls)
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) (Amazon S3 access)
- [jupyter](https://jupyter.org/) (to run the notebook)

### Install

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### AWS credentials (optional)

The full notebook reads training data from an Amazon S3 bucket. If you have access, set your own credentials as environment variables. **Never write credentials into the code or commit them to GitHub.**

```bash
export AWS_ACCESS_KEY_ID=<your key>
export AWS_SECRET_ACCESS_KEY=<your secret>
export AWS_SESSION_TOKEN=<your token>   # only for temporary credentials
export AWS_DEFAULT_REGION=us-east-1
```

No AWS account? Use the local sample dataset in [Get Started](#get-started) instead.

## Get Started

Quickly train the demo on a small dataset (no AWS needed). The sample file [`data/sample_gas_prices.csv`](data/sample_gas_prices.csv) has two columns: `Week_Ending` (date) and `Regular` (average regular gas price in dollars).

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load the small sample dataset
data = pd.read_csv("data/sample_gas_prices.csv", parse_dates=["Week_Ending"])
data = data.sort_values("Week_Ending")

# 2. Turn dates into numbers (days since the first week)
data["day_number"] = (data["Week_Ending"] - data["Week_Ending"].min()).dt.days

# 3. Train the model
X = data[["day_number"]]
y = data["Regular"]
model = LinearRegression()
model.fit(X, y)
print(f"Gas Price = {model.intercept_:.2f} + ({model.coef_[0]:.4f} x Day Number)")
print(f"R-squared: {model.score(X, y):.2f}")

# 4. Predict a future date
future = pd.to_datetime("2027-01-03")
day = (future - data["Week_Ending"].min()).days
price = model.predict(pd.DataFrame({"day_number": [day]}))[0]
print(f"Predicted price on {future.date()}: ${price:.2f}")
```

**Expected output:** the fitted regression equation, the R-squared value, and a predicted price in dollars per gallon for the chosen date.

### Run the full pipeline

Open the notebook and run the cells from top to bottom:

```bash
jupyter notebook Initial_Demo_Final_Code.ipynb
```

The notebook also prints 52 weeks of future predictions, saves them to a CSV for ArcGIS, and graphs actual prices, the regression line, and future predictions. See [`bin/run_example.py`](bin/run_example.py) for a complete example script.

## Project Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── Initial_Demo_Final_Code.ipynb   # full demo notebook
├── src/                            # reusable model and data code
├── data/sample_gas_prices.csv      # small dataset for quick start
├── docs/documentation.md           # detailed function documentation
└── bin/run_example.py              # example of how to run the code
```

## Documentation

Detailed documentation of every major function (inputs, outputs, how it works) is in [`docs/documentation.md`](docs/documentation.md). Summary of the current pipeline:

| Step | What it does | Inputs / Outputs |
|------|--------------|------------------|
| Ingest current data | Calls the GitHub API for the [US-Gas-Prices](https://github.com/lykmapipo/US-Gas-Prices) repo, finds the newest metro daily-average CSV, and downloads it | In: API URL. Out: DataFrame |
| Filter to Charlotte | Keeps rows where `State-Abbreviation` is NC and `Metro-Name` contains "Charlotte" | In: all-metro DataFrame. Out: Charlotte DataFrame |
| Upload to S3 | Writes the Charlotte data to the S3 bucket under `raw-data/` | In: DataFrame. Out: CSV in S3 |
| Load training data | Reads weekly processed price history from S3 | In: bucket and key. Out: DataFrame |
| Prepare features | Converts dates, sorts, and creates `day_number` (days since first week) | In: raw DataFrame. Out: `X`, `y` |
| Train model | Fits scikit-learn `LinearRegression`, prints equation and R-squared | In: `X`, `y`. Out: trained model |
| Predict | Predicts the next 52 weeks, or one user-entered date | In: dates. Out: predicted $/gallon |
| Export for ArcGIS | Saves predictions with Charlotte latitude and longitude to CSV | In: predictions. Out: CSV |
| Plot | Graphs actual prices, regression line, and future predictions | In: data. Out: matplotlib figure |

## Planned Classes

These planned extensions will support the cost savings by gas station distance index. They are **not yet implemented**.

### `GasStation`

Represents one station with attributes `station_id`, `brand`, and `location`.

- `get_price_history()`: retrieves historical prices for that station
- `calculate_avg_price()`: calculates its average price over a certain period
- `calculate_price_change()`: measures how much its price has recently increased or decreased
- `compare_to_area_average()`: determines whether the station is normally cheaper or more expensive than nearby stations

### `DemandSector`

Represents gas demand by residential, commercial, or industrial sector.

- `sector_id`: unique identifier for the sector
- `sector_name`: descriptive name (e.g., "Residential", "Heavy Industrial", "Commercial")
- `average_consumption_rate`: baseline volume of gas typically consumed per hour or day
- `peak_demand_hours`: times of day when the sector hits maximum usage
- `volatility_index`: how much the sector's demand fluctuates unexpectedly

## Examples

We include three examples so users can see the tool under both ideal and realistic conditions and can check its accuracy before relying on forecasts:

1. **Major metro county (Mecklenburg/Charlotte):** a typical forecast use case with plenty of historical data.
2. **Rural county with sparse data (planned):** shows how the model handles limited historical records.
3. **Predicted vs. actual prices (planned):** compares predictions to real prices over a past time window to demonstrate accuracy.

## Contact

Questions? Contact **[Student Name]** at [email@charlotte.edu](mailto:email@charlotte.edu), or open an [Issue](../../issues) on this repository.

lmaddula@uncc.edu
rrober45@charlotte.edu
rrahman4@charlotte.edu
mabunura@charlotte.edu

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
