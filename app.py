import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Infosys Data Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load Data
df = pd.read_csv("infosys.csv")

# Load Model
model = joblib.load("model.pkl")

# Title
st.title("📊 Infosys Quarterly Revenue Analytics Dashboard")

st.markdown("---")

# KPI Section
col1, col2, col3 = st.columns(3)

latest_revenue = df["Revenue"].iloc[-1]
avg_revenue = df["Revenue"].mean()

growth = (
    (df["Revenue"].iloc[-1] - df["Revenue"].iloc[-2])
    / df["Revenue"].iloc[-2]
) * 100

col1.metric("Latest Revenue", f"₹{latest_revenue:,.0f} Cr")
col2.metric("Average Revenue", f"₹{avg_revenue:,.0f} Cr")
col3.metric("Growth Rate", f"{growth:.2f}%")

st.markdown("---")

# Revenue Trend Chart
st.subheader("📈 Quarterly Revenue Trend")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    df["Quarter"],
    df["Revenue"],
    marker="o",
    linewidth=3
)

ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue (Cr)")
ax.set_title("Infosys Revenue Trend")

st.pyplot(fig)

# Historical Data
st.subheader("📋 Historical Revenue Data")

st.dataframe(
    df,
    use_container_width=True
)

st.markdown("---")

# Prediction Section
st.subheader("🔮 Next Quarter Prediction")

quarter = st.number_input(
    "Enter Future Quarter Number",
    min_value=1,
    value=len(df) + 1
)

if st.button("Predict Revenue"):

    prediction = model.predict([[quarter]])[0]

    st.success(
        f"Predicted Revenue for Quarter {quarter}: ₹{prediction:,.2f} Cr"
    )

    # Forecast Graph
    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.plot(
        df["Quarter"],
        df["Revenue"],
        marker="o",
        linewidth=3,
        label="Historical Revenue"
    )

    ax2.scatter(
        quarter,
        prediction,
        s=150,
        label="Predicted Revenue"
    )

    ax2.set_xlabel("Quarter")
    ax2.set_ylabel("Revenue (Cr)")
    ax2.set_title("Revenue Forecast")

    ax2.legend()

    st.pyplot(fig2)

st.markdown("---")

st.subheader("📊 Revenue Statistics")

st.write(df["Revenue"].describe())