# Atmospheric State Transition Predictor 🌦️
**Advanced Machine Learning Project: Hidden Markov Models (HMM)**

## 📌 Project Overview
This project implements a **Hidden Markov Model (HMM)** to predict weather state transitions. Unlike standard classifiers, an HMM assumes that the system is a Markov process with unobserved (hidden) states. This model uses the **Viterbi Algorithm** to predict the most likely sequence of future weather conditions based on current observations.

## 🚀 Features
* **Transition Matrix Visualization:** See the probability of moving from one weather state to another.
* **Viterbi Sequence Prediction:** Forecasts the most likely weather "path" for the next 7 days.
* **Interactive Dashboard:** Built with Streamlit for real-time user input.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **ML Library:** `hmmlearn` (for HMM implementation)
* **Data Handling:** `NumPy` & `Pandas`
* **UI Framework:** `Streamlit`

## 📉 How it Works
The model defines:
1.  **Hidden States:** The actual weather (Sunny, Rainy, Cloudy).
2.  **Observations:** Recorded events (e.g., Humidity levels or Dry/Wet ground).
3.  **Transition Probabilities:** The "memory" of the model—how likely it is to stay sunny vs. turn rainy.

## 🏃 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt