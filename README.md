# 🍁 Canada Per Capita Income Predictor

A premium, interactive web application that leverages a **Simple Linear Regression** model to predict and analyze the historical trend of Canada's per capita income (spanning 1970 to 2016).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8503)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

* **Real-time Predictor**: Forecast per capita income for any target year (1970 to 2050) using interactive slider controls.
* **Interactive Charting**: A customized Plotly visualization that renders actual historical data points, the regression line, and updates the targeted forecast coordinate in real-time.
* **Mathematical Breakdowns**: Displays the underlying Y-Intercept ($\beta_0$) and Coefficient/Slope ($\beta_1$) parameters, alongside LaTeX equations explaining the ordinary least squares implementation.
* **Model Evaluation Dashboard**: Shows statistical performance metrics like the $R^2$ (Coefficient of Determination), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
* **Historical Data Explorer**: Allows users to search and filter through raw dataset records, review key descriptive statistics (mean, std dev, min/max), and download the full dataset as a CSV.
* **Premium Aesthetics**: Engineered with a responsive dark-themed glassmorphism interface, custom typography, glowing UI highlights, and smooth hover effects.

---

## 🔬 Mathematical Formulation

The dashboard fits a simple linear regression equation where the **dependent variable** ($Income$) is modeled as a linear function of the **independent variable** ($Year$):

$$\text{Income} = \beta_1 \cdot \text{Year} + \beta_0$$

### Model Coefficents:
* **Growth Rate ($\beta_1$)**: Approximately **$828.47** per year.
* **Y-Intercept ($\beta_0$)**: **-$1,632,210.76** (theoretical value at Year 0).

---

## 🛠️ Tech Stack & Requirements

* **Python 3.8+**
* **Streamlit** (UI Framework)
* **Scikit-Learn** (Machine Learning Model)
* **Plotly** (Interactive Graphics)
* **Pandas** & **NumPy** (Data Management)

---

## 🚀 Quick Start Guide

### 1. Clone the repository:
```bash
git clone https://github.com/mrgraciz123/CANADA-PER-CAPITA-INCOME-PREDICTION.git
cd CANADA-PER-CAPITA-INCOME-PREDICTION
```

### 2. Install Dependencies:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### 3. Run the App locally:
```bash
python -m streamlit run app.py
```
Open your browser and navigate to the local URL (usually `http://localhost:8501`).

---

## 📁 Repository Structure

```
├── .gitignore                          # Ignored caches & Jupyter checkpoint files
├── 2_outliers_z_score_Question.ipynb   # Supplemental outlier analytics notebook
├── app.py                              # Main Streamlit application entrypoint
├── canada_income_prediction.ipynb      # Linear regression training notebook
├── canada_per_capita_income.csv        # Historical per capita income dataset
└── README.md                           # Project documentation
```
