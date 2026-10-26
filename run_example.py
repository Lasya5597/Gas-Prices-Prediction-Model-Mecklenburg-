#!/usr/bin/env python3
"""
run_example.py
==============
End-to-end example: train the Charlotte gas price model on a small local CSV,
print the equation, predict one date, forecast the next year, and save the
results. No AWS account needed.

Usage (run from the repository root):

    python bin/run_example.py
    python bin/run_example.py --data data/sample_gas_prices.csv --date 2027-06-06
    python bin/run_example.py --weeks 26 --output-dir results
"""

import argparse
import os
import sys

# Make "src" importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gas_price_model import (  # noqa: E402
    export_predictions,
    forecast_weeks,
    load_local_csv,
    plot_forecast,
    predict_price,
    prepare_features,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Charlotte gas price demo")
    parser.add_argument("--data", default="data/sample_gas_prices.csv",
                        help="CSV with Week_Ending and Regular columns")
    parser.add_argument("--date", default="2027-01-03",
                        help="Single future date to predict (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=52,
                        help="Number of future weeks to forecast")
    parser.add_argument("--output-dir", default="output",
                        help="Folder for the forecast CSV and graph")
    args = parser.parse_args()

    # 1. Load data
    try:
        raw = load_local_csv(args.data)
    except FileNotFoundError:
        sys.exit(f"Could not find {args.data}. Add a sample CSV with columns "
                 "Week_Ending and Regular, or pass --data <path>.")
    print(f"Loaded {len(raw)} weekly rows from {args.data}")

    # 2. Prepare features and train
    data, X, y = prepare_features(raw)
    model = train_model(X, y)
    print(f"\nGas Price = {model.intercept_:.2f} + ({model.coef_[0]:.4f} x Day Number)")
    print(f"R-squared: {model.score(X, y):.2f}")

    # 3. Predict one date
    price = predict_price(model, data, args.date)
    print(f"\nPredicted average Regular price for {args.date}: ${price:.2f} per gallon")

    # 4. Forecast the next N weeks
    forecast = forecast_weeks(model, data, periods=args.weeks)
    print(f"\nFirst 5 of {args.weeks} forecast weeks:")
    print(forecast.head().to_string(index=False))

    # 5. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "gas_price_predictions.csv")
    png_path = os.path.join(args.output_dir, "gas_price_forecast.png")
    export_predictions(forecast, csv_path)
    plot_forecast(data, model, forecast, save_path=png_path)
    print(f"\nSaved forecast CSV to {csv_path}")
    print(f"Saved graph to {png_path}")


if __name__ == "__main__":
    main()
