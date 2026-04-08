import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model_logic import train_weather_hmm, predict_next_state

# UI Configuration
st.set_page_config(page_title="Weather HMM Predictor", page_icon="🌤️")

st.title("🌦️ Atmospheric State Transition Predictor")
st.markdown("### Advanced Machine Learning: Hidden Markov Model (HMM)")

# Initialize Model from your model_logic.py file
model, states = train_weather_hmm()

# Sidebar Setup
st.sidebar.header("Control Panel")
current_weather = st.sidebar.selectbox("Current Weather State:", states)
forecast_days = st.sidebar.slider("Forecast Horizon (Days):", 1, 10, 5)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 State Transition Matrix")
    df_trans = pd.DataFrame(model.transmat_, index=states, columns=states)
    
    fig, ax = plt.subplots()
    sns.heatmap(df_trans, annot=True, cmap="Blues", ax=ax)
    plt.title("Probability of State Change")
    st.pyplot(fig)
    st.caption("Rows: 'Today' | Columns: 'Tomorrow'")

with col2:
    st.subheader("🔮 Multi-Day Forecast")
    current_idx = states.index(current_weather)
    
    if st.button("Run Prediction"):
        results = predict_next_state(model, current_idx, forecast_days)
        res_names = [states[r] for r in results]
        
        forecast_df = pd.DataFrame({
            "Future Day": [f"Day {i+1}" for i in range(forecast_days)],
            "Predicted State": res_names
        })
        
        st.table(forecast_df)
        st.success(f"Forecast generated starting from {current_weather}!")

st.divider()
st.markdown("Explanation: This HMM uses a **Transition Matrix** to calculate the most likely next state.")