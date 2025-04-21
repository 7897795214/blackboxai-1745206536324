from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load models and feature columns
lr_model = joblib.load('Gemini/models/linear_regression_model.pkl')
rf_model = joblib.load('Gemini/models/random_forest_model.pkl')
feature_columns = joblib.load('Gemini/models/feature_columns.pkl')

# Load and preprocess data for dropdowns
data_path = 'retail_store_inventory 2.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
products = df['Product ID'].unique()
stores = df['Store ID'].unique()

# HTML template as a string
template = """
<!DOCTYPE html>
<html lang="en" >
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gemini - Inventory Demand Forecasting</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background-color: #f9fafb;
      color: #1f2937;
    }
    .container {
      max-width: 900px;
      margin: 2rem auto;
      background-color: #ffffff;
      border-radius: 0.5rem;
      padding: 2rem;
      box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    h1 {
      color: #2563eb;
    }
    label {
      color: #374151;
    }
    select, input[type="file"], button, input[type="number"] {
      background-color: #f3f4f6;
      color: #1f2937;
      border: 1px solid #d1d5db;
    }
    select:focus, input[type="file"]:focus, button:focus, input[type="number"]:focus {
      outline: none;
      border-color: #2563eb;
      box-shadow: 0 0 5px #2563eb;
    }
    button {
      transition: background-color 0.3s ease;
    }
    button:hover {
      background-color: #1d4ed8;
      color: white;
    }
    #result {
      margin-top: 1rem;
      font-size: 1.25rem;
      font-weight: 600;
      color: #2563eb;
    }
    #extra-info {
      margin-top: 1rem;
      background-color: #e0e7ff;
      padding: 1rem;
      border-radius: 0.5rem;
      color: #1e40af;
      font-size: 1rem;
      line-height: 1.5;
    }
    canvas {
      margin-top: 2rem;
      background-color: #f3f4f6;
      border-radius: 0.5rem;
      padding: 1rem;
    }
    #loadingOverlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(255, 255, 255, 0.7);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 9999;
      display: none;
    }
    .spinner {
      border: 6px solid #e5e7eb;
      border-top: 6px solid #2563eb;
      border-radius: 50%;
      width: 50px;
      height: 50px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg);}
      100% { transform: rotate(360deg);}
    }
  </style>
</head>
<body>
  <div id="loadingOverlay">
    <div class="spinner"></div>
  </div>
  <div class="container">
    <h1 class="text-3xl font-bold text-center mb-6">Gemini - Inventory Demand Forecasting</h1>
    <form id="forecastForm" class="space-y-6" enctype="multipart/form-data">
      <div>
        <label for="store" class="block mb-2">Select Store</label>
        <select id="store" name="store" required class="w-full rounded px-3 py-2">
          {% for store in stores %}
          <option value="{{ store }}">{{ store }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="product" class="block mb-2">Select Product</label>
        <select id="product" name="product" required class="w-full rounded px-3 py-2">
          {% for product in products %}
          <option value="{{ product }}">{{ product }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="model" class="block mb-2">Select Model</label>
        <select id="model" name="model" required class="w-full rounded px-3 py-2">
          <option value="Linear Regression">Linear Regression</option>
          <option value="Random Forest">Random Forest</option>
        </select>
      </div>
      <div>
        <label for="dataset" class="block mb-2">Upload Dataset (CSV)</label>
        <input type="file" id="dataset" name="dataset" accept=".csv" class="w-full rounded px-3 py-2" />
      </div>
      <div>
        <label for="forecast_horizon" class="block mb-2">Forecast Horizon (days)</label>
        <input type="number" id="forecast_horizon" name="forecast_horizon" min="1" max="30" value="7" class="w-full rounded px-3 py-2" />
      </div>
      <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded">
        Forecast Demand
      </button>
    </form>
    <div id="result" class="text-center"></div>
    <div id="extra-info" class="hidden"></div>
    <canvas id="forecastChart" width="800" height="400" class="hidden"></canvas>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const loadingOverlay = document.getElementById('loadingOverlay');
    const ctx = document.getElementById('forecastChart').getContext('2d');
    let forecastChart = null;

    document.getElementById('forecastForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      loadingOverlay.style.display = 'flex';

      const store = document.getElementById('store').value;
      const product = document.getElementById('product').value;
      const model = document.getElementById('model').value;
      const datasetInput = document.getElementById('dataset');
      const forecastHorizon = document.getElementById('forecast_horizon').value;
      const formData = new FormData();

      formData.append('store_id', store);
      formData.append('product_id', product);
      formData.append('model', model);
      formData.append('forecast_horizon', forecastHorizon);
      if (datasetInput.files.length > 0) {
        formData.append('dataset', datasetInput.files[0]);
      }

      const resultDiv = document.getElementById('result');
      const extraInfoDiv = document.getElementById('extra-info');
      const forecastCanvas = document.getElementById('forecastChart');

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const data = await response.json();
          resultDiv.textContent = `Forecasted Demand (next ${forecastHorizon} days): ${data.prediction} units`;

          // Show extra info
          extraInfoDiv.classList.remove('hidden');
          extraInfoDiv.innerHTML = `
            <p><strong>Model Used:</strong> ${data.model}</p>
            <p><strong>Mean Squared Error:</strong> ${data.mse.toFixed(2)}</p>
            <p><strong>R-squared:</strong> ${data.r2.toFixed(2)}</p>
          `;

          // Show chart with bar and line datasets
          forecastCanvas.classList.remove('hidden');
          if (forecastChart) {
            forecastChart.destroy();
          }
          forecastChart = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: data.forecast_dates,
              datasets: [
                {
                  type: 'bar',
                  label: 'Forecasted Demand',
                  data: data.forecast_values,
                  backgroundColor: '#2563eb',
                },
                {
                  type: 'line',
                  label: 'Forecast Trend',
                  data: data.forecast_values,
                  borderColor: '#1e40af',
                  borderWidth: 2,
                  fill: false,
                  tension: 0.3,
                  pointRadius: 4,
                  pointHoverRadius: 6,
                }
              ]
            },
            options: {
              responsive: true,
              scales: {
                x: {
                  title: {
                    display: true,
                    text: 'Date',
                    color: '#374151'
                  },
                  ticks: {
                    color: '#374151'
                  },
                  grid: {
                    color: '#e5e7eb'
                  }
                },
                y: {
                  title: {
                    display: true,
                    text: 'Demand',
                    color: '#374151'
                  },
                  ticks: {
                    color: '#374151'
                  },
                  grid: {
                    color: '#e5e7eb'
                  }
                }
              },
              plugins: {
                legend: {
                  labels: {
                    color: '#374151'
                  }
                }
              }
            }
          });
        } else {
          const error = await response.json();
          resultDiv.textContent = `Error: ${error.error}`;
          extraInfoDiv.classList.add('hidden');
          forecastCanvas.classList.add('hidden');
        }
      } catch (error) {
        resultDiv.textContent = 'Error: Unable to fetch forecast.';
        extraInfoDiv.classList.add('hidden');
        forecastCanvas.classList.add('hidden');
      } finally {
        loadingOverlay.style.display = 'none';
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(template, products=products, stores=stores)

if __name__ == '__main__':
    if not os.path.exists('Gemini/models'):
        os.makedirs('Gemini/models')
    app.run(host='0.0.0.0', port=8080, debug=True)
