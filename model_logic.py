import numpy as np
from hmmlearn import hmm

def train_weather_hmm():
    states = ["Sunny", "Cloudy", "Rainy"]
    n_states = len(states)

    # Probabilities
    start_probability = np.array([0.6, 0.3, 0.1])
    
    transition_probability = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])

    emission_probability = np.array([
        [0.9, 0.05, 0.05],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ])

    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    return model, states

def predict_next_state(model, current_state_idx, n_steps=5):
    curr_state = current_state_idx
    predictions = []
    
    for _ in range(n_steps):
        # We pick 0, 1, or 2 based on the probabilities for the current state
        next_state = np.random.choice(3, p=model.transmat_[curr_state])
        predictions.append(next_state)
        curr_state = next_state
        
    return predictions