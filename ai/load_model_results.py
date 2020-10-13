import keras
import tensorflow as tf
import numpy as np


def load_model():
    # load json and create model
    json_file = open('models/online_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/online_model.h5")
    print("Loaded model from disk")
    return loaded_model


# Summarize model performance
def cards_to_vectors(cards:list, deck:tuple):
    # Create some helper functions for the main program loop
    n_cards = len(set(deck))
    P = np.array([[0] * n_cards])
    D = np.array([[0] * n_cards])
    for card in cards[0]:
       P[:, card - 1] += 1
    for card in cards[1]:
       D[:, card - 1] += 1
    return np.concatenate((P, D), axis=1)
  

def get_predctions():
    model = load_model()
    deck = (1,2,3,4,5,6,7,8,9,10,11)
    action = np.zeros((10, 10, 10))
    for i in range(2, 12):
        for j in range(2, 12): # Dealer up card
            for k in range(2, 12):  # Dealer up card
                #if j < i:
                #    action[i-2][j-2][k-2] = 0
                #else:
                cards = cards_to_vectors([[i, j], [k]], deck)
                q_values = model.predict(cards)[0]
                action[i-2][j-2][k-2] = q_values[0:3].sum()
    for k in range(action.shape[2]):
        action[9,9,k] = 0
    return action
