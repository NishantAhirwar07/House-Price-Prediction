# House Price Prediction Model

This project demonstrates a basic Machine Learning workflow using **Linear Regression** to predict house prices based on area. It provides a foundational implementation for understanding how predictive modeling works in practice.

---

## 📋 Overview
The objective of this project is to model the relationship between house area and price using a simple dataset. It utilizes Python's data analysis and machine learning ecosystem to process, visualize, and predict outcomes based on historical data.

## 🛠 Prerequisites

Ensure you have Python installed on your system. You will need the following libraries to execute the script:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>
   ```
2. **Prepare the Data:**
   Ensure the dataset file `homeprices (1).csv` is located in the root directory of the project.
3. **Execute the Script:**
   ```bash
   python house_price_prediction.py
   ```

## 📊 Project Logic

The project follows a standard Machine Learning pipeline:

* **Data Loading**: Imports the CSV dataset using `pandas` to prepare the features (`area`) and target (`price`).
* **Visualization**: Uses `matplotlib` to perform exploratory data analysis (EDA) and visualize the linear relationship between variables.
* **Model Training**:
    * Splits data into training (80%) and testing (20%) sets using `train_test_split`.
    * Uses `LinearRegression` from `scikit-learn` to fit the best-fit line defined by the equation $y = mx + c$.
* **Prediction**: The script outputs the calculated slope ($m$) and intercept ($c$), allowing you to calculate the price for any specific area input.

## 📁 Project Structure

* `house_price_prediction.py`: The primary Python script containing the ML training and evaluation logic.
* `homeprices (1).csv`: The source dataset containing area and price records.
* `README.md`: Project documentation.
