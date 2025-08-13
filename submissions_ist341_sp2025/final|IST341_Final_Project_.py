# install darts



# install u8darts[all]


from darts import TimeSeries

series = TimeSeries.from_csv(
    "Vegas_Traffic_Tidy_Validated.csv",
    time_col="Date",
    value_cols="Visitor Volume",
    fill_missing_dates=True,
    freq="MS"
)



train, val = series.split_before(0.85)



from darts.models import ARIMA

model = ARIMA()
model.fit(train)
forecast = model.predict(len(val))

series.plot(label="actual")
forecast.plot(label="forecast")



from darts.models import RNNModel

rnn = RNNModel(model='RNN', input_chunk_length=12, output_chunk_length=6, n_epochs=300)
rnn.fit(train)
forecast_rnn = rnn.predict(len(val))

series.plot(label="actual")
forecast_rnn.plot(label="RNN forecast")



from darts.dataprocessing.transformers import Scaler

scaler = Scaler()
series_scaled = scaler.fit_transform(series)

train, val = series_scaled.split_before(0.85)



from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

# Scale the data
scaler = Scaler()
series_scaled = scaler.fit_transform(series)
train, val = series_scaled.split_before(0.85)

# Train a forecasting model
model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42
)

model.fit(train)

# Predict and invert scale
forecast = model.predict(len(val))
forecast_inv = scaler.inverse_transform(forecast)

# Plot
series.plot(label="actual")
forecast_inv.plot(label="forecast (N-BEATS)")



import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")

# Create time series
series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Gaming Revenue : Clark County", fill_missing_dates=True, freq="MS")

# Normalize
scaler = Scaler()
series_scaled = scaler.fit_transform(series)
# Convert the string to a pandas Timestamp before dropping
# Use the actual end date of the series as the split point
train = series_scaled.drop_after(pd.Timestamp("2024-12-01"))

# Train model
model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=500, random_state=42)
model.fit(train)

# Forecast for 2025
forecast = model.predict(12)
forecast_unscaled = scaler.inverse_transform(forecast)

# Plot
series.plot(label="Historical")
forecast_unscaled.plot(label="Forecast 2025")
plt.title("Clark County Gaming Revenue Forecast (2025)")
plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.utils import generate_index # Import generate_index
import torch

# Load data
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")

# Preprocess
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Extract target time series
series = TimeSeries.from_series(df['Visitor Volume'])

# Optionally add datetime covariates (month, year, etc.) - Note: NBEATS does not support future_covariates directly during fit.
# If you want to use covariates with NBEATS, you would typically include them in the input data for training.
# However, the error indicates future_covariates is not supported for fit.
year_series = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
month_series = datetime_attribute_timeseries(series, attribute="month", one_hot=True)
covariates = year_series.stack(month_series)

# Train/val split
train, val = series.split_after(pd.Timestamp("2023-12-01"))
cov_train, cov_val = covariates.split_after(pd.Timestamp("2023-12-01"))

# Define and train the model
model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},  # or "gpu" if available
)

# Fit the model without future_covariates as it is not supported by this model
model.fit(train, verbose=True)

# Forecast 12 months (2025)
# When predicting with future covariates, you need to provide them.
# The covariates should cover the forecast horizon.
# Ensure `covariates` covers the period for which you are forecasting (12 months after the end of `series`).
# In this case, `future_cov` should be the covariates for the 12 months you want to predict.
# A common approach is to generate future covariates based on the forecast dates.
# For example:
future_cov_dates = generate_index(start=series.end_time() + series.freq, length=12, freq=series.freq)
future_cov = datetime_attribute_timeseries(future_cov_dates, attribute="year", one_hot=False).stack(
             datetime_attribute_timeseries(future_cov_dates, attribute="month", one_hot=True))

forecast = model.predict(n=12)

# Plot
series.plot(label="Actual")
forecast.plot(label="Forecast")
plt.title("Visitor Volume Forecast for 2025")
plt.legend()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import LightGBMModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import generate_index
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

# Define the target variable
series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Prepare external regressors
# -------------------------------
exog_cols = [
    "Convention Attendance",
    "Total En/Deplaned Passengers",
    "Total Room Nights Occupied",
    "Available Room Inventory",
    "Avg. Daily Auto Traffic: All Major Highways*"
]

# Select and forward-fill missing values
df_exog = df[exog_cols].copy().fillna(method="ffill")

# Extend future covariates to cover 2025 by repeating the last known row
last_row = df_exog.iloc[-1]
future_exog = pd.DataFrame([last_row] * 12, columns=exog_cols)
future_dates = pd.date_range(start="2025-01-01", periods=12, freq="MS")
future_exog.index = future_dates

# Combine existing and future covariates
combined_exog = pd.concat([df_exog, future_exog])
full_covariates = TimeSeries.from_dataframe(
    combined_exog,
    fill_missing_dates=True,
    freq="MS"
)

# -------------------------------
# 3. Train/test split
# -------------------------------
# Last available real data is December 2024
train = series.drop_after(pd.Timestamp("2024-12-01"))
train_cov = full_covariates.drop_after(pd.Timestamp("2024-12-01"))

# Forecasting for 12 months (2025)
future_cov = full_covariates.slice(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-01"))

# -------------------------------
# 4. Define and train the model
# -------------------------------
model = LightGBMModel(lags=12, lags_future_covariates=[1])
model.fit(series=train, future_covariates=train_cov)

# -------------------------------
# 5. Forecast
# -------------------------------
forecast = model.predict(n=12, future_covariates=future_cov)

# -------------------------------
# 6. Plot and export
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with External Regressors")
plt.legend()
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import LightGBMModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

# Define target series
series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Economic external regressors
# -------------------------------
exog_cols = [
    "Convention Attendance",
    "Total En/Deplaned Passengers",
    "Total Room Nights Occupied",
    "Available Room Inventory",
    "Avg. Daily Auto Traffic: All Major Highways*"
]

# Forward-fill missing values
df_exog = df[exog_cols].copy().fillna(method="ffill")

# Extend into 2025 by repeating last known values
last_row = df_exog.iloc[-1]
future_exog = pd.DataFrame([last_row] * 12, columns=exog_cols)
future_dates = pd.date_range(start="2025-01-01", periods=12, freq="MS")
future_exog.index = future_dates

# Combine for full regressor coverage
combined_exog = pd.concat([df_exog, future_exog])
economic_cov = TimeSeries.from_dataframe(combined_exog, freq="MS")

# -------------------------------
# 3. One-hot encoded month covariates
# -------------------------------
# Generate full time index into 2025
full_index = pd.date_range(start=series.start_time(), periods=len(series) + 12, freq="MS")

# Generate month one-hot encodings
month_series = datetime_attribute_timeseries(
    TimeSeries.from_times_and_values(full_index, [0] * len(full_index)),
    attribute="month",
    one_hot=True
)

# -------------------------------
# 4. Combine all covariates
# -------------------------------
full_covariates = economic_cov.stack(month_series)

# Split into training and future periods
train = series.drop_after(pd.Timestamp("2024-12-01"))
train_cov = full_covariates.drop_after(pd.Timestamp("2024-12-01"))
future_cov = full_covariates.slice(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-01"))

# -------------------------------
# 5. Train LightGBM model
# -------------------------------
model = LightGBMModel(lags=12, lags_future_covariates=[1])
model.fit(series=train, future_covariates=train_cov)

# -------------------------------
# 6. Forecast 2025
# -------------------------------
forecast = model.predict(n=12, future_covariates=future_cov)

# -------------------------------
# 7. Plot and export
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with Regressors + Seasonality")
plt.legend()
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Train/test split
# -------------------------------
train = series.drop_after(pd.Timestamp("2024-12-01"))

# -------------------------------
# 3. Define and train model
# -------------------------------
model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},  # change to 'gpu' if available
)

model.fit(train, verbose=True)

# -------------------------------
# 4. Predict 12 months (2025)
# -------------------------------
forecast = model.predict(n=12)

# -------------------------------
# 5. Plot and save
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with N-BEATS")
plt.legend()
plt.tight_layout()
plt.show()


print("Train ends at:", train.end_time())
forecast = model.predict(n=12)
print("Forecast starts at:", forecast.start_time())



print(model.model)



# Get forecast values and timestamps
values = forecast.values().flatten()
dates = forecast.time_index

# Print first 3 forecast values with their dates
print("Forecast for first 3 months of 2025:")
for date, value in zip(dates[:3], values[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value):,} visitors")


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Train/test split
# -------------------------------
train = series[:pd.Timestamp("2024-12-01")]

# -------------------------------
# 3. Define and train model
# -------------------------------
model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},  # change to 'gpu' if available
)

model.fit(train, verbose=True)

# -------------------------------
# 4. Predict 12 months (2025)
# -------------------------------
forecast = model.predict(n=12)

# -------------------------------
# 5. Plot and save
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with N-BEATS")
plt.legend()
plt.tight_layout()
plt.show()


print("Train ends at:", train.end_time())
forecast = model.predict(n=12)
print("Forecast starts at:", forecast.start_time())


# Get forecast values and timestamps
values = forecast.values().flatten()
dates = forecast.time_index

# Print first 3 forecast values with their dates
print("Forecast for first 3 months of 2025:")
for date, value in zip(dates[:3], values[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value):,} visitors")


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Train/test split
# -------------------------------
train = series[:pd.Timestamp("2024-12-01")]

# -------------------------------
# 3. Define and train model
# -------------------------------
model = NBEATSModel(
    input_chunk_length=12,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},  # change to 'gpu' if available
)

model.fit(train, verbose=True)

# -------------------------------
# 4. Predict 12 months (2025)
# -------------------------------
forecast = model.predict(n=12)

# -------------------------------
# 5. Plot and save
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with N-BEATS")
plt.legend()
plt.tight_layout()
plt.show()


# Get forecast values and timestamps
values = forecast.values().flatten()
dates = forecast.time_index

# Print first 3 forecast values with their dates
print("Forecast for first 3 months of 2025:")
for date, value in zip(dates[:3], values[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value):,} visitors")


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Train/test split
# -------------------------------
train = series[:pd.Timestamp("2024-12-01")]

# -------------------------------
# 3. Define and train model
# -------------------------------
model = NBEATSModel(
    input_chunk_length=36,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},  # change to 'gpu' if available
)

model.fit(train, verbose=True)

# -------------------------------
# 4. Predict 12 months (2025)
# -------------------------------
forecast = model.predict(n=12)

# -------------------------------
# 5. Plot and save
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with N-BEATS")
plt.legend()
plt.tight_layout()
plt.show()


# Get forecast values and timestamps
values = forecast.values().flatten()
dates = forecast.time_index

# Print first 3 forecast values with their dates
print("Forecast for first 3 months of 2025:")
for date, value in zip(dates[:3], values[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value):,} visitors")


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 2. Train/test split
# -------------------------------
train = series[:pd.Timestamp("2024-12-01")]

# -------------------------------
# 3. Define and train model
# -------------------------------
model = NBEATSModel(
    input_chunk_length=60,
    output_chunk_length=12,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},  # change to 'gpu' if available
)

model.fit(train, verbose=True)

# -------------------------------
# 4. Predict 12 months (2025)
# -------------------------------
forecast = model.predict(n=12)

# -------------------------------
# 5. Plot and save
# -------------------------------
series.plot(label="Actual")
forecast.plot(label="Forecast (2025)")
plt.title("Visitor Volume Forecast with N-BEATS")
plt.legend()
plt.tight_layout()
plt.show()


# Get forecast values and timestamps
values = forecast.values().flatten()
dates = forecast.time_index

# Print first 3 forecast values with their dates
print("Forecast for first 3 months of 2025:")
for date, value in zip(dates[:3], values[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value):,} visitors")





# -------------------------------
# 🔧 2. Imports and setup
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
import torch
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 📈 3. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 🔄 4. Normalize the series
# -------------------------------
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# -------------------------------
# 🧠 5. Define and train RNN model
# -------------------------------
input_chunk_length = 24
training_length = 24
forecast_horizon = 12

# Train on data through Dec 2024
train = series_scaled[:pd.Timestamp("2024-12-01")]

model = RNNModel(
    model="RNN",  # or "LSTM", "GRU"
    input_chunk_length=input_chunk_length,
    training_length=training_length,
    hidden_dim=64,
    n_rnn_layers=2,
    dropout=0.2,
    batch_size=16,
    n_epochs=1500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
)

model.fit(train, verbose=True)

# -------------------------------
# 🔮 6. Forecast 2025
# -------------------------------
forecast_scaled = model.predict(n=forecast_horizon)
forecast = scaler.inverse_transform(forecast_scaled)

# -------------------------------
# 🖨️ 7. Print forecast
# -------------------------------
print("Forecast for first 3 months of 2025:")
for date, value in zip(forecast.time_index[:3], forecast.values()[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value[0]):,} visitors")

# -------------------------------
# 📊 8. Plot forecast
# -------------------------------
if forecast.start_time() <= series.end_time():
    forecast = forecast.drop_before(series.end_time() + series.freq)

from darts import TimeSeries as TS
combined = TS.concatenate(series, forecast)

combined.plot(label="Actual + Forecast")
plt.axvline(x=series.end_time(), color="gray", linestyle="--", label="Forecast Start")
plt.title("Visitor Volume Forecast with RNN")
plt.legend()
plt.tight_layout()
plt.show()





# -------------------------------
# 🔧 2. Imports and setup
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 📈 3. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")
series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 🔄 4. Normalize the series
# -------------------------------
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# -------------------------------
# 🧠 5. Define early stopping
# -------------------------------
early_stop = EarlyStopping(
    monitor="train_loss",
    patience=10,
    mode="min"
)

# -------------------------------
# 🧠 6. Define and train RNN model
# -------------------------------
input_chunk_length = 71
training_length = 71
forecast_horizon = 12

# Train on data through Dec 2024
train = series_scaled[:pd.Timestamp("2024-12-01")]

model = RNNModel(
    model="GRU",  # or "LSTM", "GRU" RNN
    input_chunk_length=input_chunk_length,
    training_length=training_length,
    hidden_dim=64,
    n_rnn_layers=2,
    dropout=0.2,
    batch_size=16,
    n_epochs=300,
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "callbacks": [early_stop],  # ✅ Early stopping included
    },
)

model.fit(train, verbose=True)

# -------------------------------
# 🔮 7. Forecast 2025
# -------------------------------
forecast_scaled = model.predict(n=forecast_horizon)
forecast = scaler.inverse_transform(forecast_scaled)

# -------------------------------
# 🖨️ 8. Print forecast
# -------------------------------
print("Forecast for first 3 months of 2025:")
for date, value in zip(forecast.time_index[:3], forecast.values()[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value[0]):,} visitors")

# -------------------------------
# 📊 9. Plot forecast
# -------------------------------
from darts import TimeSeries as TS
if forecast.start_time() <= series.end_time():
    forecast = forecast.drop_before(series.end_time() + series.freq)

combined = TS.concatenate(series, forecast)

combined.plot(label="Actual + Forecast")
plt.axvline(x=series.end_time(), color="gray", linestyle="--", label="Forecast Start")
plt.title("Visitor Volume Forecast with RNN + Early Stopping")
plt.legend()
plt.tight_layout()
plt.show()





# -------------------------------
# 📦 1. Install packages
# -------------------------------

# -------------------------------
# 🔧 2. Imports and setup
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import torch
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 📈 3. Load and prepare data
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values("Date").set_index("Date")

series = TimeSeries.from_series(df["Visitor Volume"])

# -------------------------------
# 🔄 4. Normalize the target
# -------------------------------
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# -------------------------------
# 📅 5. Create future covariates (one-hot month)
# -------------------------------
# Generate full index including 2025
full_index = pd.date_range(start=series.start_time(), periods=len(series) + 12, freq="MS")
full_series = TimeSeries.from_times_and_values(full_index, [0] * len(full_index))

month_covariates = datetime_attribute_timeseries(
    full_series, attribute="month", one_hot=True
)

# -------------------------------
# ✂️ 6. Train/test split
# -------------------------------
train_series = series_scaled[:pd.Timestamp("2024-12-01")]
train_covariates = month_covariates[:pd.Timestamp("2024-12-01")]
future_covariates = month_covariates.slice(
    pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-01")
)

# -------------------------------
# 🧠 7. Define and train TFT
# -------------------------------
model = TFTModel(
    input_chunk_length=24,
    output_chunk_length=12,
    hidden_size=32,
    lstm_layers=1,
    dropout=0.1,
    batch_size=16,
    n_epochs=300,
    random_state=42,
    likelihood=None,
    pl_trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu"},
)

model.fit(train_series, future_covariates=train_covariates, verbose=True)

# -------------------------------
# 🔮 8. Forecast
# -------------------------------
forecast_scaled = model.predict(n=12, future_covariates=future_covariates)
forecast = scaler.inverse_transform(forecast_scaled)

# -------------------------------
# 🖨️ 9. Print forecast
# -------------------------------
print("Forecast for first 3 months of 2025:")
for date, value in zip(forecast.time_index[:3], forecast.values()[:3]):
    print(f"{date.strftime('%B %Y')}: {int(value[0]):,} visitors")

# -------------------------------
# 📊 10. Plot results
# -------------------------------
from darts import TimeSeries as TS
if forecast.start_time() <= series.end_time():
    forecast = forecast.drop_before(series.end_time() + series.freq)

combined = TS.concatenate(series, forecast)

combined.plot(label="Actual + Forecast")
plt.axvline(x=series.end_time(), color="gray", linestyle="--", label="Forecast Start")
plt.title("Visitor Volume Forecast with TFT (Month Seasonality)")
plt.legend()
plt.tight_layout()
plt.show()


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Load the cleaned dataset
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")

# Select relevant columns and drop missing values
df = df[["Total Room Nights Occupied", "Gaming Revenue : Clark County"]].dropna()

# Prepare features and target
X = df[["Total Room Nights Occupied"]]
y = df["Gaming Revenue : Clark County"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Regressor
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=["Total Room Nights Occupied"], filled=True, rounded=True)
plt.title("Decision Tree: Predicting Gaming Revenue from Room Nights Occupied")
plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Set up model and scorer
rf = RandomForestRegressor(random_state=42)
scorer = make_scorer(r2_score)

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=scorer, n_jobs=-1)
grid_search.fit(X, y)

# Output the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated R² score:", grid_search.best_score_)



best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error
print("Test R²:", r2_score(y_test, y_pred))
print("Test MAE:", mean_absolute_error(y_test, y_pred))



import matplotlib.pyplot as plt

importances = best_rf.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance in Random Forest")
plt.show()


features = [
    "Total Room Nights Occupied",
    "Available Room Inventory",
    "Visitor Volume",
    "Convention Attendance",
    "Average Daily Room Rate (ADR)"
]

df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df = df[features + ["Gaming Revenue : Clark County"]].dropna()

X = df[features]
y = df["Gaming Revenue : Clark County"]



# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")

# Select multiple relevant features
features = [
    "Total Room Nights Occupied",
    "Available Room Inventory",
    "Visitor Volume",
    "Convention Attendance",
    "Average Daily Room Rate (ADR)",
    "Avg. Daily Auto Traffic: All Major Highways*"
]

# Define target
target = "Gaming Revenue : Clark County"

# Drop missing values
df = df[features + [target]].dropna()

# Split features and target
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.0f}")

# Plot feature importances
importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="green")
plt.xlabel("Importance")
plt.title("Feature Importance in Random Forest")
plt.tight_layout()
plt.show()


from sklearn.tree import plot_tree

# Pick one tree from the forest (e.g., the first one)
one_tree = best_rf.estimators_[0]

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(
    one_tree,
    feature_names=features,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Sample Tree from Optimized Random Forest")
plt.show()



import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Set target and covariates
target_col = "Visitor Volume"
covariate_cols = [
    "Convention Attendance",
    "Total Room Nights Occupied",
    "Total En/Deplaned Passengers",
    "Avg. Daily Auto Traffic: All Major Highways*"
]

# Create TimeSeries objects
target_series = TimeSeries.from_dataframe(df, 'Date', target_col)
covariates_series = TimeSeries.from_dataframe(df, 'Date', covariate_cols)

# Scale data
scaler_target = Scaler()
scaler_covs = Scaler()

target_scaled = scaler_target.fit_transform(target_series)
covs_scaled = scaler_covs.fit_transform(covariates_series)

# Split: last 12 months for testing
train_target, val_target = target_scaled[:-12], target_scaled[-12:]
train_covs, val_covs = covs_scaled[:-12], covs_scaled[-12:]

# Define and train the RNN model
model = RNNModel(
    model='RNN',
    input_chunk_length=24,
    output_chunk_length=12,
    hidden_dim=25,
    n_rnn_layers=1,
    dropout=0.1,
    batch_size=16,
    n_epochs=300,
    random_state=42,
    model_name="visitor_volume_rnn",
    force_reset=True
)

model.fit(train_target, future_covariates=train_covs, verbose=True)


# Predict
forecast = model.predict(n=12, future_covariates=covs_scaled)


# Inverse transform to original scale
forecast = scaler_target.inverse_transform(forecast)
actual = scaler_target.inverse_transform(val_target)

# Plot
target_series.plot(label="Actual (Full)")
forecast.plot(label="Forecast")
actual.plot(label="Actual (Last 12)")
plt.title("Visitor Volume Forecast with RNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# install darts[torch]


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import generate_index
import matplotlib.pyplot as plt

# -------------------------------
# LOAD & CLEAN DATA
# -------------------------------
df = pd.read_csv("Vegas_Traffic_Tidy_Validated.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Filter: remove 2020–2021 and limit training to Dec 2024
df_filtered = df[
    (df.index.year != 2020) &
    (df.index.year != 2021) &
    (df.index < "2025-01-01")
]

# -------------------------------
# SELECT TARGET AND COVARIATES
# -------------------------------
target_col = "Visitor Volume"
covariate_cols = [
    "Convention Attendance",
    "Total Room Nights Occupied",
    "Total En/Deplaned Passengers",
    "Avg. Daily Auto Traffic: All Major Highways*"
]

# -------------------------------
# CONVERT TO DARTS TimeSeries
# -------------------------------
series = TimeSeries.from_series(df_filtered[target_col], fill_missing_dates=True, freq="MS")
covariates = TimeSeries.from_dataframe(df_filtered[covariate_cols], fill_missing_dates=True, freq="MS")

# -------------------------------
# SCALE TARGET AND COVARIATES
# -------------------------------
scaler_series = Scaler()
scaler_covariates = Scaler()

series_scaled = scaler_series.fit_transform(series)
covariates_scaled = scaler_covariates.fit_transform(covariates)

# -------------------------------
# DEFINE AND TRAIN RNN MODEL
# -------------------------------
model = RNNModel(
    model="RNN",
    input_chunk_length=12,
    n_epochs=300,
    random_state=42,
    model_name="rnn_vegas_forecast",
    dropout=0.1,
    training_length=24,
    log_tensorboard=False,
    force_reset=True
)

model.fit(series_scaled, future_covariates=covariates_scaled)

# -------------------------------
# CREATE FUTURE COVARIATES (Jan 2024 – Dec 2026)
# -------------------------------
future_dates = pd.date_range(start="2024-01-01", periods=36, freq="MS")

# Use last known covariate values
last_values = covariates_scaled[-1].all_values().reshape(-1)

# Create DataFrame of repeated covariates
future_data = pd.DataFrame(
    [last_values] * 36,
    columns=covariates_scaled.components,
    index=future_dates
)

# Convert to TimeSeries
future_covariates = TimeSeries.from_dataframe(future_data, fill_missing_dates=True, freq="MS")

# -------------------------------
# FORECAST 24 MONTHS INTO 2026
# -------------------------------
forecast = model.predict(24, future_covariates=future_covariates)
forecast = scaler_series.inverse_transform(forecast)

# Confirm forecast range
print("Forecast starts:", forecast.start_time())
print("Forecast ends:  ", forecast.end_time())

# -------------------------------
# PLOT RESULTS
# -------------------------------
plt.figure(figsize=(12, 6))
series.plot(label="Historical")
forecast.plot(label="Forecast (2025–2026)")
plt.title("Visitor Volume Forecast (RNN + Future Covariates)\nExcludes 2020–2021")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))
series.plot(label="Historical")
forecast.plot(label="Forecast (2025–2026)")
plt.title("Visitor Volume Forecast (RNN + Future Covariates)\nExcludes 2020–2021")
plt.grid(True)
plt.legend()
plt.xlim([pd.Timestamp("2019-01-01"), pd.Timestamp("2026-12-31")])  # <-- force full range
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()


import pandas as pd

# Get forecast values explicitly
forecast_values = forecast.all_values().flatten()  # Ensures 1D array

# Create DataFrame
forecast_df = pd.DataFrame({
    "date": forecast.time_index,
    "visitor_volume": forecast_values
}).set_index("date")

print(forecast_df)


test_forecast = model.predict(1, future_covariates=future_covariates)

print("Test forecast values:", test_forecast.all_values())
print("Test forecast start:", test_forecast.start_time())
print("Test forecast end:", test_forecast.end_time())



