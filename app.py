from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(page_title="Logistics AI Dashboard", layout="wide")

DATA_PATHS = [
    Path("DataCoSupplyChainDataset.csv"),
    Path("compressed_data.csv.gz"),
]
DATASET_HINT = " or ".join(path.name for path in DATA_PATHS)

@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        dataset_path = next((path for path in DATA_PATHS if path.exists()), None)
        if dataset_path is None:
            raise FileNotFoundError

        df = pd.read_csv(dataset_path, encoding="latin-1")

        df['Delay_Duration'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

        cols_to_keep = {
            'Shipping Mode': 'Shipping_Mode',
            'Order City': 'Destination',
            'Customer City': 'Origin',
            'Order Region': 'Region',
            'Delay_Duration': 'Delay_Duration'
        }

        existing_cols = [c for c in cols_to_keep.keys() if c in df.columns]
        df = df[existing_cols].rename(columns=cols_to_keep)

        df = df.dropna()

        return df

    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: Dataset not found. Expected {DATASET_HINT}.")
        st.info("Please download the dataset from Kaggle and place it in the same folder as app.py")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

df = load_data()
st.title("🚚 Real-World Supply Chain: Delay Prediction")
st.markdown("A Machine Learning dashboard predicting shipping delays using the **DataCo Smart Supply Chain** dataset.")



@st.cache_resource
def train_model(dataframe: pd.DataFrame):
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()
    le_mode = LabelEncoder()

    dataframe = dataframe.copy()
    dataframe['Origin_Code'] = le_origin.fit_transform(dataframe['Origin'])
    dataframe['Dest_Code'] = le_dest.fit_transform(dataframe['Destination'])
    dataframe['Mode_Code'] = le_mode.fit_transform(dataframe['Shipping_Mode'])

    X = dataframe[['Origin_Code', 'Dest_Code', 'Mode_Code']]
    y = dataframe['Delay_Duration']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return model, le_origin, le_dest, le_mode, mae


model, le_origin, le_dest, le_mode, mae = train_model(df)



col1, col2, col3 = st.columns(3)
col1.metric("Total Orders Analyzed", f"{len(df):,}")
col2.metric("Avg Global Delay", f"{df['Delay_Duration'].mean():.2f} Days")
col3.metric("Model Accuracy (MAE)", f"{mae:.2f} Days", help="Average error in days.")

tab1, tab2 = st.tabs(["📊 Analytics Overview", "🤖 Live Prediction Tool"])

with tab1:
    st.subheader("Historical Performance")
    
    c1, c2 = st.columns(2)
    with c1:
        avg_delay_mode = df.groupby('Shipping_Mode')['Delay_Duration'].mean().reset_index()
        fig_mode = px.bar(avg_delay_mode, x='Shipping_Mode', y='Delay_Duration', 
                         title="Average Delay by Shipping Mode", color='Delay_Duration', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_mode, use_container_width=True)
        
    with c2:
        fig_hist = px.histogram(df, x='Delay_Duration', nbins=20, 
                               title="Distribution of Delays (Days)", color_discrete_sequence=['#3366CC'])
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Regional Analysis")
    top_delay_cities = df.groupby('Destination')['Delay_Duration'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_cities = px.bar(top_delay_cities, x='Delay_Duration', y='Destination', orientation='h',
                        title="Top 10 Cities with Highest Average Delays")
    st.plotly_chart(fig_cities, use_container_width=True)

with tab2:
    st.sidebar.header("Predict New Shipment")
    st.subheader("Predict Delay for a New Order")
    
    top_origins = df['Origin'].value_counts().head(50).index
    top_dests = df['Destination'].value_counts().head(50).index
    
    input_origin = st.selectbox("Origin City", sorted(top_origins))
    input_dest = st.selectbox("Destination City", sorted(top_dests))
    input_mode = st.selectbox("Shipping Mode", le_mode.classes_)
    
    if st.button("Predict Delay"):
        try:
            if input_origin in le_origin.classes_ and input_dest in le_dest.classes_:
                in_orig_enc = le_origin.transform([input_origin])[0]
                in_dest_enc = le_dest.transform([input_dest])[0]
                in_mode_enc = le_mode.transform([input_mode])[0]
                
                input_data = pd.DataFrame([[in_orig_enc, in_dest_enc, in_mode_enc]], 
                                        columns=['Origin_Code', 'Dest_Code', 'Mode_Code'])
                
                prediction = model.predict(input_data)[0]
                
                st.divider()
                st.metric("Predicted Delay", f"{prediction:.2f} Days")
                
                if prediction > 2.0:
                    st.error("⚠️ Risk: Significant delay expected.")
                elif prediction > 0:
                    st.warning("⚠️ Risk: Minor delay expected.")
                else:
                    st.success("✅ Good News: Expected to arrive on time or early.")
            else:
                st.warning("Selected city not found in training data (LabelEncoder error).")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
