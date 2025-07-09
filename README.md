# F1 Lap Time Prediction with Deep Learning

A machine learning project that predicts Formula 1 lap times using telemetry data and metadata from Max Verstappen's performances at Silverstone (British Grand Prix) from 2021-2025.

## 🏎️ Project Overview

This project combines real-time telemetry data (Speed, Throttle, Brake) with race metadata (tire compound, tire life, weather conditions) to predict lap times using LSTM neural networks. The data spans multiple years and session types to capture various racing conditions.

## 📊 Dataset

- **Driver**: Max Verstappen (VER)
- **Circuit**: Silverstone (British Grand Prix)
- **Years**: 2021-2025
- **Sessions**: FP1, FP2, FP3, Qualifying, Race
- **Data Sources**: FastF1 API

### Features Used

- **Telemetry Data**: Speed, Throttle, Brake (150 time steps @ 0.1s resolution)
- **Metadata**: Tire compound, tire life, air temperature, track temperature
- **Target**: Lap time (seconds)

## 🛠️ Project Structure

```text
f1/
├── data_preparation.ipynb    # Data collection and preprocessing
├── data_merging.ipynb       # Feature engineering and data combination
├── model_training.ipynb     # LSTM model training and evaluation
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/aayushxtech/F1-laptime-prediction.git
cd F1-laptime-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows   source venv/bin/activate on Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Data Preparation (`data_preparation.ipynb`)

- Collects lap data from FastF1 API
- Filters for quick laps only
- Merges weather data with lap information
- Extracts telemetry sequences (15-second windows)
- Normalizes and processes data

#### 2. Data Merging (`data_merging.ipynb`)

- One-hot encodes tire compounds
- Normalizes numerical features
- Combines telemetry and metadata
- Creates final training datasets

#### 3. Model Training (`model_training.ipynb`)

- Implements LSTM architecture
- Trains on 70% of data, validates on 15%, tests on 15%
- Uses StandardScaler for target normalization
- Saves trained model weights

## 🧠 Model Architecture

### LSTM Neural Network

- Input: Combined telemetry + metadata (150 timesteps × 6 features)
- Hidden layers: 4 LSTM layers with 128 hidden units
- Output: Single regression value (predicted lap time)
- Loss function: Mean Squared Error
- Optimizer: Adam (lr=0.001)

## 📈 Results

The model achieves excellent prediction accuracy with very low error margins. For example, on test case lap index 10:
![image](https://github.com/user-attachments/assets/cf598678-1ce8-479d-a959-4e8c94b837b2)


- **Actual Lap Time**: 98.393 seconds
- **Predicted Lap Time**: 97.394 seconds
- **Error**: 0.999 seconds (~1% difference)

The model successfully learns patterns from:

- **Driving behavior**: Throttle, brake, and speed patterns throughout the lap
- **Track conditions**: Temperature effects on tire and car performance
- **Tire performance**: Compound selection and degradation over stint length
- **Session context**: Different driving styles between practice, qualifying, and race conditions

### Model Performance Metrics

- **Mean Absolute Error**: < 1.0 seconds
- **Training Loss**: Converged to ~0.28 after 30 epochs
- **Validation Loss**: Stabilized around 0.32
- **Prediction Accuracy**: 98%+ for most test cases

## 🔧 Key Technologies

- **FastF1**: F1 data acquisition and telemetry processing
- **PyTorch**: Deep learning framework for LSTM implementation
- **scikit-learn**: Data preprocessing and train/test splits
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib**: Data visualization

## 📝 Data Pipeline

1. **Collection**: FastF1 API → Raw session data
2. **Filtering**: Quick laps only → Quality control
3. **Telemetry**: 15s windows @ 0.1s resolution → Uniform sequences
4. **Weather**: Time-aligned temperature data → Environmental context
5. **Engineering**: Normalization + encoding → ML-ready features
6. **Training**: LSTM regression → Lap time prediction

## ⚡ Performance Considerations

- **Memory**: Telemetry data can be large; consider batch processing
- **Compute**: GPU acceleration recommended for training
- **Cache**: FastF1 caches data locally for faster subsequent runs

## 🙏 Acknowledgments

- **FastF1**: For providing excellent F1 data access
- **Formula 1**: For the exciting sport that makes this analysis possible
- **Max Verstappen**: For consistently fast lap times to predict!

---

Happy Racing! 🏁
