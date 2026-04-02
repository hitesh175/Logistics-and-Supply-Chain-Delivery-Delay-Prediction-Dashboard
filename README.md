# Logistics-and-Supply-Chain-Delivery-Delay-Prediction-Dashboard
The dashboard allows stakeholders to input shipment details via a sidebar and immediately receive a risk assessment score and estimated delay duration in the main view, enabling proactive decision-making.

Here is a professional `README.md` file for your **Logistics & Supply Chain** project. You can create a new file named `README.md` in your project folder and paste this content directly into it.

#  Supply Chain: Delivery Delay Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Scikit-Learn](https://img.shields.io/badge/ML-RandomForest-orange)

##  Overview

This project is an end-to-end Machine Learning application designed to optimize logistics operations. It utilizes historical supply chain data to predict the likelihood and duration of shipping delays.

**The dashboard allows stakeholders to input shipment details via a sidebar and immediately receive a risk assessment score and estimated delay duration in the main view, enabling proactive decision-making.**

##  Features

*  Live Prediction Tool: Interactive sidebar for stakeholders to input route details (Origin, Destination, Mode) and get instant delay predictions.
*  Analytics Dashboard: Visualizes historical performance, including average delays by shipping mode and regional bottlenecks.
*  Risk Assessment: Automatically categorizes shipments as "Safe," "Minor Risk," or "High Risk" based on predicted delay duration.
*  Real-World Data: Built using the **DataCo Smart Supply Chain** dataset to model realistic logistics scenarios.

##  Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly Express

##  How to Run Locally

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### 2. Installation
Install the required dependencies:
```bash
pip install pandas numpy scikit-learn streamlit plotly

```

### 3. Dataset Setup

This project requires the **DataCo Smart Supply Chain for Big Data Analysis** dataset.

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
2. Rename the file to `DataCoSupplyChainDataset.csv`.
3. Place it in the root directory of the project (same folder as `app.py`).

### 4. Run the Application

Launch the dashboard using Streamlit:

```bash
streamlit run app.py

```

##  Project Structure

```text
├── app.py                       # Main application script (ETL, Modeling, Dashboard)
├── DataCoSupplyChainDataset.csv # Dataset (Required)
├── README.md                    # Project Documentation
└── requirements.txt             # List of dependencies

```

##  Model Logic

The application uses a **Random Forest Regressor** to predict the `Delay_Duration` target variable.

* **Target Calculation:** `Real Shipping Days` - `Scheduled Shipping Days`.
* **Features Used:** Origin City, Destination City, and Shipping Mode.
* **Encoding:** Label Encoding is used to convert categorical city names into numerical format for the model.

##  Contact

Created by **Hitesh Kanagala Rajendra Prasad**.

* [LinkedIn Profile](https://www.linkedin.com/in/hitesh-k-r-544973111/)
* [GitHub Profile](https://github.com/hitesh175)

```

```
