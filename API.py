from flask import Flask, jsonify, send_file
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
from datetime import timedelta
import requests

app = Flask(__name__)


@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        # Fetch data
        url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
        response = requests.get(url)
        response.raise_for_status()

        # Parse data
        lines = response.text.splitlines()
        data_start_index = next(i for i, line in enumerate(lines) if not line.startswith('#'))
        data = pd.read_csv(io.StringIO('\n'.join(lines[data_start_index:])), sep='\s+', header=None)

        # Extract relevant columns
        dates = pd.to_datetime(data[[0, 1, 2]].rename(columns={0: 'Year', 1: 'Month', 2: 'Day'}))
        ap_data = data.iloc[:, 23]

        # Process recent data
        recent_ap_values = ap_data.iloc[-8:-1].values
        arima_model = ARIMA(recent_ap_values, order=(2, 1, 2))
        arima_fitted = arima_model.fit()
        arima_forecast = np.clip(np.round(arima_fitted.forecast(steps=7)).astype(int), 0, None)

        # Generate forecast dates
        last_date = pd.Timestamp(dates.iloc[-2])
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        forecast_dates_formatted = [date.strftime('%Y-%m-%d') for date in forecast_dates]

        # Send data as JSON
        return jsonify({
            "forecast_dates": forecast_dates_formatted,
            "arima_forecast": arima_forecast.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph', methods=['GET'])
def graph():
    # Create the graph
    fig, ax = plt.subplots()
    dates = ["2024-12-30", "2024-12-31"]
    values = [20, 30]
    ax.plot(dates, values, label="Mock Data")
    ax.legend()
    ax.grid()

    # Save the graph to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)

    # Send the graph as a PNG image
    return send_file(buf, mimetype='image/png')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
