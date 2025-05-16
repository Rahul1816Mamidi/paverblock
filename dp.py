import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data and train model once ---
@st.cache_data
def load_data_and_train():
    df = pd.read_csv(r"C:\Users\bhavy\OneDrive\Desktop\Rahul\dpdataset.csv")

    X = df[['LDPE', 'Kaolin', 'Temp']]
    y = df['Compressive_Mpa']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    threshold = df['Compressive_Mpa'].quantile(0.25)
    return df, model, y, y_pred, threshold

df, model, y, y_pred, threshold = load_data_and_train()

# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "prediction"

# --- Page Navigation functions ---
def go_to_graphs():
    st.session_state.page = "graphs"

def go_to_prediction():
    st.session_state.page = "prediction"

# --- Prediction Page ---
def prediction_page():
    st.title("🧱 Paver Block Performance Predictor")
    st.markdown("Enter the composition details to predict paver block performance.")
    st.subheader("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        ldpe = st.selectbox("LDPE Content (%)", list(range(5, 16)), index=5)
        kaolin = st.slider("Kaolin Content (g)", 0, 10, step=1)
        sand = st.slider("Sand Content (%)", 70, 90, step=1, value=85)

    with col2:
        temp = st.slider("Curing Temperature (°C)", 100, 200, step=5, value=125)
        mix_time = st.number_input("Mixing Time (min)", value=30)
        cool_time = st.number_input("Cooling Time (min)", value=60)

    if st.button("Predict Performance"):
        input_data = pd.DataFrame([[ldpe, kaolin, temp]], columns=['LDPE', 'Kaolin', 'Temp'])
        prediction = model.predict(input_data)[0]
        melting_point = round(133 + (temp / 100 * 5), 1)
        risk = "Low" if prediction > threshold else "High"
        risk_color = "green" if risk == "Low" else "red"

        st.markdown("---")
        st.subheader("Prediction Results")
        col1_out, col2_out = st.columns(2)

        with col1_out:
            st.markdown(f"*Compressive Strength:* {prediction:.2f} MPa")
            st.markdown(f"*Failure Risk:* :{risk_color}[{risk}]")
            st.markdown(f"*Risk Threshold:* {threshold:.2f} MPa")
            st.markdown(f"*Melting Point:* {melting_point}°C")

        with col2_out:
            st.markdown("*Input Summary*")
            st.markdown(f"- LDPE: {ldpe}%  \n- Kaolin: {kaolin}g  \n- Sand: {sand}%")
            st.markdown(f"- Curing Temp: {temp}°C  \n- Mixing Time: {mix_time} min  \n- Cooling Time: {cool_time} min")

        st.markdown("---")
        st.subheader("Analysis & Recommendations")
        st.markdown("*Prediction Summary*")
        st.markdown(f"- Compressive Strength: {prediction:.1f} MPa  \n- Failure Risk: {risk}  \n- Estimated Melting Point: {melting_point}°C")

        st.markdown("*Recommendations:*")
        recs = []
        if ldpe >= 10:
            recs.append("✅ Current LDPE proportion ensures good flexibility.")
        else:
            recs.append("⚠ Consider increasing LDPE for flexibility.")
        if kaolin > 5:
            recs.append("⚠ Kaolin is slightly high — consider decreasing to improve bonding.")
        else:
            recs.append("✅ Kaolin level is optimal for strength.")
        if 120 <= temp <= 140:
            recs.append("✅ Curing temperature is optimal for polymer fusion.")
        else:
            recs.append("⚠ Adjust curing temperature to 120–140°C for best performance.")
        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("*Explanation:*")
        st.markdown("""
        The prediction is based on how LDPE, Kaolin, and temperature affect material bonding and strength.
        Lower predicted compressive strength than threshold indicates high failure risk.
        """)

    if st.button("📊 Show Graphs"):
        go_to_graphs()

# --- Graphs Page ---
def graphs_page():
    st.title("📊 Paver Block Performance Graphs")

    st.subheader("Actual vs Predicted Compressive Strength")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y, y=y_pred, ax=ax1)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax1.set_xlabel("Actual Strength (MPa)")
    ax1.set_ylabel("Predicted Strength (MPa)")
    ax1.set_title("Actual vs Predicted")
    st.pyplot(fig1)

    st.subheader("Accuracy Metrics")
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    st.write(f"*R² Score:* {r2:.2f}")
    st.write(f"*Mean Squared Error:* {mse:.2f}")

    fig2, ax2 = plt.subplots()
    metrics = ['R² Score', 'MSE']
    values = [r2, mse]
    ax2.bar(metrics, values, color=['blue', 'orange'])
    ax2.set_title("Model Accuracy Metrics")
    st.pyplot(fig2)

    st.subheader("Failure Risk Distribution")
    high_risk_count = (y <= threshold).sum()
    low_risk_count = (y > threshold).sum()
    risk_counts = [high_risk_count, low_risk_count]
    labels = ['High Risk', 'Low Risk']
    colors = ['red', 'green']

    fig3, ax3 = plt.subplots()
    ax3.pie(risk_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Failure Risk Distribution in Dataset")
    st.pyplot(fig3)

    if st.button("🏠 Back to Home"):
        go_to_prediction()

# --- Main ---
if st.session_state.page == "prediction":
    prediction_page()
elif st.session_state.page == "graphs":
    graphs_page()