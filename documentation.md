# Documentation: Charlotte Gas Price Predictor

This document extends the [README](../README.md). It explains what each major function does, how it works, and how to call it. Everything below lives in [`src/gas_price_model.py`](../src/gas_price_model.py), and a runnable end-to-end example is in [`bin/run_example.py`](../bin/run_example.py).

## Contents

- [Overview](#overview)
- [The model](#the-model)
- [Quick example](#quick-example)
- [Function reference](#function-reference)
  - [1. Getting data](#1-getting-data): `fetch_latest_metro_data`, `filter_metro`, `upload_to_s3`, `load_from_s3`, `load_local_csv`
  - [2. Preparing data and training](#2-preparing-data-and-training): `prepare_features`, `train_model`
  - [3. Predicting](#3-predicting): `predict_price`, `forecast_weeks`
  - [4. Output](#4-output): `export_predictions`, `plot_forecast`
- [Command-line example](#command-line-example)
- [Data format](#data-format)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Planned classes](#planned-classes)

---

## Overview

| | |
|---|---|
| **Goal** | Forecast the average Regular gas price for the Charlotte, NC metro area |
| **Model** | Simple linear regression (scikit-learn `LinearRegression`) |
| **Input** | Weekly average Regular prices (`Week_Ending`, `Regular`) |
| **Output** | A predicted price for one date, or a 52-week forecast with a graph and an ArcGIS-ready CSV |
| **Data sources** | [US-Gas-Prices](https://github.com/lykmapipo/US-Gas-Prices) (via GitHub API), Amazon S3, or any local CSV |

The pipeline:

```
fetch / load data  ->  prepare_features  ->  train_model  ->  predict_price / forecast_weeks  ->  export_predictions / plot_forecast
```

## The model

The model fits a straight line through historical prices:

```
Gas Price = intercept + slope x day_number
```

- `day_number` is the number of days since the first week in the training data. Regression models need numbers, not dates, so `prepare_features()` converts each date this way.
- The **intercept** is the model's estimated price on day 0. The **slope** is the estimated change in price per day.
- **R-squared** (from `model.score(X, y)`) tells you how much of the variation in historical prices the line explains. 1.0 is a perfect fit and 0 means the line is no better than the average price.

## Quick example

```python
from src.gas_price_model import (
    load_local_csv, prepare_features, train_model, predict_price, forecast_weeks
)

raw = load_local_csv("data/sample_gas_prices.csv")
data, X, y = prepare_features(raw)
model = train_model(X, y)

print(predict_price(model, data, "2027-01-03"))   # one date -> float
print(forecast_weeks(model, data, periods=12))    # next 12 weeks -> DataFrame
```

---

## Function reference

### 1. Getting data

#### `fetch_latest_metro_data(api_url=DEFAULT_API_URL)`

Downloads the newest metro daily-average CSV from the GitHub API.

**How it works:** it asks the GitHub API for the list of files in the data folder, keeps only `.csv` files, and sorts them by filename. Because the filenames contain dates, the last one is the newest. It then downloads that file and reads it into a DataFrame.

| Parameter | Type | Description |
|---|---|---|
| `api_url` | `str` | GitHub API "contents" URL of the folder holding the CSVs. Defaults to the US-Gas-Prices metro folder. |

**Returns:** `pandas.DataFrame` with prices for every metro area.
**Raises:** `requests.HTTPError` if GitHub is unreachable or rate-limits the request.

```python
all_metros = fetch_latest_metro_data()
```

#### `filter_metro(df, state="NC", metro_name="Charlotte")`

Keeps only the rows for one metro area.

**How it works:** it selects rows where `State-Abbreviation` equals `state` and `Metro-Name` contains `metro_name` (case-insensitive), and returns a copy.

| Parameter | Type | Description |
|---|---|---|
| `df` | `DataFrame` | Output of `fetch_latest_metro_data()` |
| `state` | `str` | Two-letter state abbreviation |
| `metro_name` | `str` | Text to look for in `Metro-Name` |

**Returns:** `pandas.DataFrame` containing only the matching rows.

```python
charlotte = filter_metro(all_metros)                      # Charlotte, NC
raleigh = filter_metro(all_metros, "NC", "Raleigh")       # any other metro
```

#### `upload_to_s3(df, bucket, key)`

Writes a DataFrame to Amazon S3 as a CSV.

**How it works:** it converts the DataFrame to CSV text in memory and uploads it with boto3's `put_object`. Credentials come from your environment variables, never from the code.

| Parameter | Type | Description |
|---|---|---|
| `df` | `DataFrame` | Data to upload |
| `bucket` | `str` | S3 bucket name |
| `key` | `str` | File path inside the bucket, e.g. `raw-data/current_charlotte_gas_prices.csv` |

**Returns:** nothing.

```python
upload_to_s3(charlotte, "charlotte-gas-price-project-dtsc3601",
             "raw-data/current_charlotte_gas_prices.csv")
```

#### `load_from_s3(bucket, key)`

Reads a CSV from Amazon S3 into a DataFrame. This is how the original notebook loads its weekly training data.

| Parameter | Type | Description |
|---|---|---|
| `bucket` | `str` | S3 bucket name |
| `key` | `str` | Path of the CSV inside the bucket |

**Returns:** `pandas.DataFrame`.

```python
raw = load_from_s3("charlotte-gas-price-project-dtsc3601",
                   "processed-data/charlotte_nc_gas_prices_weekly.csv")
```

#### `load_local_csv(path)`

Reads a local CSV, so the demo runs without AWS.

**How it works:** it reads the file and checks that the columns `Week_Ending` and `Regular` are present.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Path to the CSV |

**Returns:** `pandas.DataFrame`.
**Raises:** `FileNotFoundError` if the file is missing, `ValueError` if a required column is missing.

---

### 2. Preparing data and training

#### `prepare_features(data)`

Converts dates to numbers so the model can use them.

**How it works:**
1. Parses `Week_Ending` as real dates.
2. Sorts from oldest to newest.
3. Adds `day_number` = days since the earliest week.
4. Splits the result into inputs (`X`) and the target (`y`).

| Parameter | Type | Description |
|---|---|---|
| `data` | `DataFrame` | Must contain `Week_Ending` and `Regular` |

**Returns:** a tuple `(data, X, y)`:
- `data`: sorted copy with the new `day_number` column. Keep this, because later functions use it.
- `X`: DataFrame with the single column `day_number` (model input).
- `y`: Series of `Regular` prices (what we predict).

```python
data, X, y = prepare_features(raw)
```

#### `train_model(X, y)`

Fits the linear regression.

**How it works:** it creates a scikit-learn `LinearRegression` and calls `.fit(X, y)`, which finds the intercept and slope that minimize squared prediction error on the training data.

| Parameter | Type | Description |
|---|---|---|
| `X` | `DataFrame` | From `prepare_features()` |
| `y` | `Series` | From `prepare_features()` |

**Returns:** the trained `LinearRegression` model.

```python
model = train_model(X, y)
print(model.intercept_, model.coef_[0])   # equation
print(model.score(X, y))                  # R-squared
```

---

### 3. Predicting

#### `predict_price(model, data, date)`

Predicts the average Regular price for a single date.

**How it works:** it converts the date to a `day_number` counted from the same starting week used in training, then passes it to the model. The `data` argument is needed so the day count starts from the same place.

| Parameter | Type | Description |
|---|---|---|
| `model` | `LinearRegression` | From `train_model()` |
| `data` | `DataFrame` | From `prepare_features()` |
| `date` | `str` or datetime | Date to predict, e.g. `"2027-01-03"` |

**Returns:** `float`, the predicted price in dollars per gallon.

```python
predict_price(model, data, "2027-01-03")   # e.g. 3.20
```

#### `forecast_weeks(model, data, periods=52, start=None)`

Predicts prices for a run of consecutive weeks (each week ends on a Sunday).

**How it works:** it builds a list of weekly Sunday dates, converts each to a `day_number`, runs the model on all of them at once, and rounds the results to cents.

| Parameter | Type | Description |
|---|---|---|
| `model` | `LinearRegression` | From `train_model()` |
| `data` | `DataFrame` | From `prepare_features()` |
| `periods` | `int` | Number of weeks to forecast (default 52) |
| `start` | `str`, optional | First forecast week. Defaults to the first Sunday after the last week in the data. |

**Returns:** `pandas.DataFrame` with columns `Week_Ending` and `Predicted_Regular_Price`.

```python
forecast = forecast_weeks(model, data, periods=26)
```

---

### 4. Output

#### `export_predictions(forecast, path, latitude=35.2271, longitude=-80.8431)`

Saves a forecast to CSV with coordinates so it can be mapped in ArcGIS.

**How it works:** it copies the forecast, adds `Latitude` and `Longitude` columns (default: center of Charlotte), and writes the CSV.

| Parameter | Type | Description |
|---|---|---|
| `forecast` | `DataFrame` | From `forecast_weeks()` |
| `path` | `str` | Where to write the CSV |
| `latitude`, `longitude` | `float` | Coordinates added to every row |

**Returns:** nothing. The output CSV has columns `Week_Ending`, `Predicted_Regular_Price`, `Latitude`, `Longitude`.

#### `plot_forecast(data, model, forecast, save_path=None, show=False)`

Graphs actual prices (dots), the fitted regression line, and the future forecast (dashed).

| Parameter | Type | Description |
|---|---|---|
| `data` | `DataFrame` | From `prepare_features()` |
| `model` | `LinearRegression` | From `train_model()` |
| `forecast` | `DataFrame` | From `forecast_weeks()` |
| `save_path` | `str`, optional | Save the figure as an image (e.g. `forecast.png`) |
| `show` | `bool` | Also open the graph in a window |

**Returns:** nothing.

---

## Command-line example

[`bin/run_example.py`](../bin/run_example.py) runs the full pipeline using the functions above. Run it from the repository root:

```bash
python bin/run_example.py
python bin/run_example.py --data data/sample_gas_prices.csv --date 2027-06-06 --weeks 26
```

| Option | Default | Description |
|---|---|---|
| `--data` | `data/sample_gas_prices.csv` | CSV with `Week_Ending` and `Regular` columns |
| `--date` | `2027-01-03` | Single future date to predict |
| `--weeks` | `52` | Number of weeks to forecast |
| `--output-dir` | `output` | Folder where the forecast CSV and graph are saved |

It prints the regression equation, R-squared, the single-date prediction, and the first five forecast weeks. It saves `gas_price_predictions.csv` and `gas_price_forecast.png` in the output folder.

## Data format

Training data needs two columns:

| Column | Type | Example |
|---|---|---|
| `Week_Ending` | date | `2025-06-01` |
| `Regular` | number (dollars per gallon) | `3.12` |

## Limitations and responsible use

- **The model only captures a straight-line trend.** It does not model seasonality (e.g., summer driving), oil-price shocks, hurricanes, refinery outages, or policy changes. Real prices can move sharply in ways a line cannot.
- **R-squared is measured on the same data the model was trained on.** It shows how well the line fits the past, not how accurate future predictions will be. A planned predicted-vs-actual example will test this on held-out data.
- **Forecasts get less reliable the further out they go.** Treat the 52-week forecast as a rough trend, not a guarantee.
- **The forecast is a metro-wide average.** It does not reflect the price at any specific station.
- **Do not make financial decisions based only on this tool.**

## Planned classes

These are planned extensions for the cost savings by gas station distance index. They are **not yet implemented**.

**`GasStation`** (attributes: `station_id`, `brand`, `location`)
- `get_price_history()`: retrieves historical prices for that station
- `calculate_avg_price()`: average price over a period
- `calculate_price_change()`: recent increase or decrease
- `compare_to_area_average()`: whether the station is normally cheaper or pricier than nearby stations

**`DemandSector`** (attributes: `sector_id`, `sector_name`, `average_consumption_rate`, `peak_demand_hours`, `volatility_index`)
- Represents gas demand by residential, commercial, or industrial sector.
