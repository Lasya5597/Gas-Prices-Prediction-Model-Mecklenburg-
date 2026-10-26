"""
gas_price_model.py
==================
Reusable functions for the Charlotte Gas Price Predictor.

This module turns the steps from Initial_Demo_Final_Code.ipynb into functions:

    data in  ->  prepare features  ->  train model  ->  predict / forecast  ->  export / plot

AWS credentials are NEVER stored in this file. boto3 reads them from your
environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and
AWS_SESSION_TOKEN if you use temporary credentials).
"""

from io import BytesIO, StringIO

import pandas as pd
from sklearn.linear_model import LinearRegression

# Default data source used by the original notebook
DEFAULT_API_URL = (
    "https://api.github.com/repos/lykmapipo/"
    "US-Gas-Prices/contents/data/metro-daily-averages"
)

# Center of Charlotte, used when exporting predictions for ArcGIS
CHARLOTTE_LAT = 35.2271
CHARLOTTE_LON = -80.8431


# ----------------------------------------------------------------------
# 1. Getting data
# ----------------------------------------------------------------------
def fetch_latest_metro_data(api_url=DEFAULT_API_URL):
    """Download the newest metro daily-average CSV from the GitHub API.

    Filenames in the source repo contain dates, so the last filename in
    sorted order is the newest file.

    Args:
        api_url (str): GitHub API "contents" URL of the folder holding the CSVs.

    Returns:
        pandas.DataFrame: Gas prices for every metro area in the newest file.
    """
    import requests  # imported here so the module works without it for local use

    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    csv_files = sorted(
        (f for f in response.json() if f["name"].endswith(".csv")),
        key=lambda f: f["name"],
    )
    latest = csv_files[-1]

    csv_response = requests.get(latest["download_url"], timeout=30)
    csv_response.raise_for_status()
    return pd.read_csv(StringIO(csv_response.text))


def filter_metro(df, state="NC", metro_name="Charlotte"):
    """Keep only the rows for one metro area.

    Args:
        df (pandas.DataFrame): Output of fetch_latest_metro_data().
        state (str): Two-letter state abbreviation, matched against the
            "State-Abbreviation" column.
        metro_name (str): Text to look for (case-insensitive) in "Metro-Name".

    Returns:
        pandas.DataFrame: A copy containing only the matching rows.
    """
    mask = (df["State-Abbreviation"] == state) & (
        df["Metro-Name"].str.contains(metro_name, case=False, na=False)
    )
    return df[mask].copy()


def upload_to_s3(df, bucket, key):
    """Write a DataFrame to Amazon S3 as a CSV file.

    Args:
        df (pandas.DataFrame): Data to upload.
        bucket (str): Name of the S3 bucket.
        key (str): Path of the file inside the bucket, e.g. "raw-data/file.csv".
    """
    import boto3

    buffer = StringIO()
    df.to_csv(buffer, index=False)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def load_from_s3(bucket, key):
    """Read a CSV file from Amazon S3 into a DataFrame.

    Args:
        bucket (str): Name of the S3 bucket.
        key (str): Path of the CSV inside the bucket.

    Returns:
        pandas.DataFrame: The CSV contents.
    """
    import boto3

    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(response["Body"].read()))


def load_local_csv(path):
    """Read a local CSV file (no AWS needed). Used for the quick-start demo.

    The file must have the columns "Week_Ending" (date) and "Regular"
    (average regular gas price in dollars per gallon).

    Args:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: The CSV contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required column is missing.
    """
    data = pd.read_csv(path)
    missing = {"Week_Ending", "Regular"} - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return data


# ----------------------------------------------------------------------
# 2. Preparing data and training
# ----------------------------------------------------------------------
def prepare_features(data):
    """Convert dates to numbers so a regression model can use them.

    Steps: parse "Week_Ending" as dates, sort oldest to newest, then add a
    "day_number" column = days since the first week in the data.

    Args:
        data (pandas.DataFrame): Must contain "Week_Ending" and "Regular".

    Returns:
        tuple: (data, X, y) where
            data (pandas.DataFrame): sorted copy with the new "day_number" column,
            X (pandas.DataFrame): the "day_number" column (model input),
            y (pandas.Series): the "Regular" column (what we predict).
    """
    data = data.copy()
    data["Week_Ending"] = pd.to_datetime(data["Week_Ending"])
    data = data.sort_values("Week_Ending").reset_index(drop=True)
    data["day_number"] = (data["Week_Ending"] - data["Week_Ending"].min()).dt.days
    return data, data[["day_number"]], data["Regular"]


def train_model(X, y):
    """Fit a simple linear regression: price = intercept + slope * day_number.

    Args:
        X (pandas.DataFrame): Model input from prepare_features().
        y (pandas.Series): Prices from prepare_features().

    Returns:
        sklearn.linear_model.LinearRegression: The trained model. Use
        model.intercept_ and model.coef_[0] for the equation, and
        model.score(X, y) for R-squared.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


# ----------------------------------------------------------------------
# 3. Predicting
# ----------------------------------------------------------------------
def predict_price(model, data, date):
    """Predict the average Regular gas price for one date.

    Args:
        model: Trained model from train_model().
        data (pandas.DataFrame): Prepared data (needs "Week_Ending"), used so
            day numbers are counted from the same starting week as training.
        date (str or datetime-like): Date to predict, e.g. "2027-01-03".

    Returns:
        float: Predicted price in dollars per gallon.
    """
    day_number = (pd.to_datetime(date) - data["Week_Ending"].min()).days
    return float(model.predict(pd.DataFrame({"day_number": [day_number]}))[0])


def forecast_weeks(model, data, periods=52, start=None):
    """Predict prices for a run of consecutive weeks (week ending on Sundays).

    Args:
        model: Trained model from train_model().
        data (pandas.DataFrame): Prepared data (needs "Week_Ending").
        periods (int): Number of weeks to forecast. Default 52 (one year).
        start (str, optional): First forecast week. Defaults to the first Sunday
            after the last week in the data.

    Returns:
        pandas.DataFrame: Columns "Week_Ending" and "Predicted_Regular_Price"
        (rounded to cents).
    """
    if start is None:
        start = data["Week_Ending"].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=periods, freq="W-SUN")
    days = (dates - data["Week_Ending"].min()).days
    predictions = model.predict(pd.DataFrame({"day_number": days}))
    return pd.DataFrame(
        {"Week_Ending": dates.date, "Predicted_Regular_Price": predictions.round(2)}
    )


# ----------------------------------------------------------------------
# 4. Output
# ----------------------------------------------------------------------
def export_predictions(forecast, path, latitude=CHARLOTTE_LAT, longitude=CHARLOTTE_LON):
    """Save a forecast to CSV with coordinates so ArcGIS can map it.

    Args:
        forecast (pandas.DataFrame): Output of forecast_weeks().
        path (str): Where to write the CSV file.
        latitude (float): Latitude added to every row. Default: Charlotte.
        longitude (float): Longitude added to every row. Default: Charlotte.
    """
    out = forecast.copy()
    out["Latitude"] = latitude
    out["Longitude"] = longitude
    out.to_csv(path, index=False)


def plot_forecast(data, model, forecast, save_path=None, show=False):
    """Graph actual prices, the fitted line, and the future forecast.

    Args:
        data (pandas.DataFrame): Prepared data from prepare_features().
        model: Trained model from train_model().
        forecast (pandas.DataFrame): Output of forecast_weeks().
        save_path (str, optional): If given, save the figure to this file (PNG).
        show (bool): If True, also open the figure in a window.
    """
    import matplotlib.pyplot as plt

    fitted = model.predict(data[["day_number"]])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(data["Week_Ending"], data["Regular"], label="Actual Prices")
    ax.plot(data["Week_Ending"], fitted, label="Regression Line")
    ax.plot(
        pd.to_datetime(forecast["Week_Ending"]),
        forecast["Predicted_Regular_Price"],
        "--",
        label="Future Predictions",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Gas Price ($)")
    ax.set_title("Charlotte Gas Prices - Linear Regression")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
