# Energy Management Data Analysis and Prediction

## Project Overview

This project focuses on analyzing energy management data and predicting total active energy consumption using various machine learning models. The dataset consists of electrical parameters, and different predictive models are implemented to improve energy forecasting accuracy.

The following steps were involved in the project:

- **Data Loading and Preprocessing**: Cleaning the dataset, handling missing values, and scaling features.
- **Model Implementation**: Training and evaluating different machine learning models.
- **Model Comparison**: Comparing models based on Mean Squared Error (MSE) and R-squared (R2) score.
- **Conclusion**: Determining the best-performing model for energy prediction.

## Libraries Used

- **NumPy**: Numerical operations
- **Pandas**: Data manipulation and preprocessing
- **Scikit-learn**: Machine learning tools for data preprocessing and model training
- **TensorFlow/Keras**: Deep learning framework used for CNN and LSTM models
- **Matplotlib**: Data visualization

## Data Description

The dataset includes various electrical parameters such as:

- **Average Current Measurements** (IL1, IL2, IL3)
- **Average Phase-to-Phase Voltages** (U12, U23, U31)
- **Average Total Active, Apparent, and Reactive Power** (kW, kVA, kVAR)
- **Total Active Energy (kWh)** *(Target Variable)*

## Steps in the Project

### 1. Data Loading, Preprocessing, and Splitting

- The dataset was cleaned by renaming columns and removing redundant headers.
- Missing values were handled using imputation.
- Features were standardized using `StandardScaler`.
- Data was split into training (80%) and testing (20%) sets.

### 2. Implemented Machine Learning Models

Three models were used for predicting total active energy consumption:

#### 1. Convolutional Neural Network (CNN)
- Captures spatial dependencies in the dataset.
- Includes a Conv1D layer, Flatten layer, and Dense layers.

#### 2. Long Short-Term Memory (LSTM) Network
- Designed to capture sequential dependencies in data.
- Uses LSTM layers followed by a Dense output layer.

#### 3. AdaBoost Regressor
- An ensemble learning method that improves prediction accuracy by combining weak learners.

### 3. Model Evaluation

Each model was evaluated using **Mean Squared Error (MSE)** and **R-squared (R2) score**. The results are summarized below:

| Model       | MSE   | R2-score |
|------------|-------|---------|
| CNN Model  | 0.194 | 0.9812  |
| LSTM Model | 0.223 | 0.9784  |
| AdaBoost   | 0.372 | 0.9640  |

### 4. Conclusion

- **CNN Model** achieved the best performance, with the highest R2-score and lowest MSE.
- **LSTM Model** also performed well, capturing sequential dependencies effectively.
- **AdaBoost Model** was effective but had a relatively lower performance compared to deep learning models.

## Future Work

- Hyperparameter tuning to further optimize model performance.
- Exploring additional features to enhance predictive accuracy.
- Implementing other deep learning architectures to improve results.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

To install the required libraries, you can use:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
