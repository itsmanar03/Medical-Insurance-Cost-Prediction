# Medical Insurance Cost Prediction 🏥

## 📝 Project Overview
This repository contains a data science project aimed at predicting individual medical insurance costs. By leveraging personal health attributes, the project builds a predictive model to estimate insurance charges, providing insights into the key drivers of healthcare expenses.


## 🚀 Key Features

* **Exploratory Data Analysis (EDA):** Comprehensive data visualization using `Seaborn` and `Matplotlib` to identify correlations between features like Age, BMI, Smoking status, and the final charges.
* **Data Preprocessing:** * Handling categorical variables through encoding.
    * Feature scaling and data cleaning.
    * Splitting the dataset into Training and Testing sets (80/20).
* **Machine Learning Modeling:** * Implementation of the **XGBoost Regressor** algorithm.
    * Hyperparameter tuning (n_estimators, learning_rate, max_depth) for optimal performance.
* **Evaluation Metrics:**
    * **R2 Score:** 90.32% (Demonstrating high predictive accuracy).
    * **Mean Absolute Error (MAE):** 2469.76.
* **Model Deployment:** The final trained model is serialized into a `.pkl` file using `Pickle` for seamless integration into production environments.


## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** * Data Manipulation: `Pandas`, `NumPy`
    * Visualization: `Matplotlib`, `Seaborn`
    * Machine Learning: `Scikit-learn`, `XGBoost`
    * Model Persistence: `Pickle`


## 📂 Project Structure
* `medical.ipynb`: Jupyter Notebook containing the full end-to-step-pipeline.
* `dataset_.csv`: The dataset used for training and testing.
* `medical_insurance_model.pkl`: The saved model for future predictions.


## 📊 Results Summary
The model successfully identifies that factors such as smoking status and BMI are significant predictors of insurance costs. 

| Metric | Result |
| :--- | :--- |
| **Model Algorithm** | XGBoost Regressor |
| **R2 Score** | **90.32%** |
| **MAE** | **2469.76** |


## ⚙️ How to Use
1. Clone the repository.
2. Install dependencies: `pip install pandas xgboost scikit-learn seaborn matplotlib`.
3. Run the `medical.ipynb` notebook to see the analysis and training process.
4. Use the `medical_insurance_model.pkl` file to make predictions on new data.
