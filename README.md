# Market Volatility Prediction using Deep Learning

## Project Description

A system for predicting short-term stock volatility based on high-frequency market data (order book and trade history).

├── Dockerfile
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
├── architecture.png
├── infer.py
├── poetry.lock
├── pyproject.toml
├── train.py
└── volatility_prediction
    ├── attention.py
    ├── data_downloader.py
    ├── dataset.py
    ├── model.py
    └── utils.py

### Why is it important?

Accurate volatility prediction is crucial for:
- Proper options pricing
- Trading position risk management
- Large order execution optimization
- Market making and liquidity provision

## Data

Data source: [Optiver Realized Volatility Prediction (Kaggle)](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data)

### Dataset Structure

1. Order Book Data (book_{train/test}.parquet):
- Best bid/ask prices for first and second levels
- Order volumes at these levels
- Timestamps with second precision

2. Trade History (trade_{train/test}.parquet):
- Execution prices
- Trade volumes
- Order count
- Timestamps

### Data Characteristics

- High-frequency updates (ticks)
- Data gaps due to inactivity periods
- Time series synchronization required
- Potential outliers and anomalies

### Potential Challenges

- Different scales of prices and volumes across instruments
- Missing data handling
- Corporate events consideration (stock splits)
- Change of size tick or price step

## Modeling Approach

### Solution Architecture

1. Data Preprocessing:
```
Raw Data -> Feature Engineering -> Normalization -> Temporal Alignment
```

2. Model:
- **Base**: RNN (GRU) with attention mechanism
- **Libraries**: PyTorch + PyTorch Lightning
- **Architecture**:
  * Input layer: Conv1D for local pattern extraction
  * Stock Attention: cross-instrument correlation handling
  * GRU: temporal dependency processing
  * Time Attention: temporal interval importance weighting
  * Output layer: volatility prediction

3. Training:
- **Optimizer**: Adam
- **Learning Rate Schedule**: ExponentialLR
- **Loss Function**: RMSPE (Root Mean Square Percentage Error)
- **Validation**: 5-fold CV across time periods

Architecture from https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/279170
![Architecture](/architecture.png)

## Production Pipeline

Data ingestion from multiple sources (market data feeds, historical databases)
Real-time streaming pipeline for live market data
Feature engineering (rolling windows, technical indicators, market regime features)
Measure performance using RMSPE
Regular daily retraining schedule
Model versioning and experiment tracking with MLflow
Model serving API with FastAPI
Data drift monitoring

Scheme will be provided later

### Final Implementation

- REST API for predictions
- Trading system integration
- Monitoring web interface
- Periodic model retraining
