import numpy as np
import tensorflow as tf
from keras import initializers
from keras.layers import Dense, Dropout
from keras.models import Model, Input
from keras.optimizers import Adam


# Create some helper functions for the main program loop
def create_card_vectors(n: int, deck: tuple, split: bool=False) -> np.array:
    """
    This function creates 'card vectors' - column vectors with 11 rows, where i-th row corresponds to playing cards
    with value i+1. The scalar value of each row represents the number of each card the player currently posses.
    :param n: integer specifying the number of 'card vectors' to return
    :param deck: tuple representing the card deck to be sampled from. Each element in the deck should be a card value.
    :return: a numpy array of shape (1,10) and sum of n. The positions of the 1 values is sampled the card values
     in the deck.
    """
    n_cards = len(set(deck))
    cardvec = np.array([[0] * n_cards])
    if split:
        assert n == 2
        val = np.random.choice(deck[1:], n, replace=True)  # Do not sample '1'
        cardvec[:, val] += 2
    else:
        val = np.random.choice(deck[1:], n, replace=True)  # Do not sample '1'
        for i in val:
            cardvec[:, i-1] += 1
    return cardvec


def dealer(player_hand, dealer_hand, deck: tuple):
    """
    This function simulates the behavior of a dealer in the game of "21". For a given input 'val', representing the
    value of a player's hand, the dealer draws randomly from a set of cards while attempting to increase the value of
    their hand to either match or exceed the player's hand without going over 21.
    :param player_hand: numpy array representing the player's current hand.
    :param dealer_hand: numpy array representing the dealer's current (revealed) hand.
    :param deck: tuple representing the card deck to be sampled from. Each element in the deck should be a card value.
    :return: a boolean value indicating whether a player won (True) or lost (False) the hand. Also, a flag
    (int) giving other information about the outcome of the round.
    """
    vals = np.unique(deck)
    assert (player_hand * vals).sum() <= 21
    while (dealer_hand * vals).sum() < (player_hand * vals).sum() and (dealer_hand * vals).sum() < 17:
        new_card = create_card_vectors(1, deck)
        dealer_hand = dealer_hand + new_card
        if (dealer_hand * vals).sum() > 21 and dealer_hand[0][10] > 0:
            dealer_hand[0][10] -= 1
            dealer_hand[0][0] += 1
    if (dealer_hand * vals).sum() > 21:
        flag = 0
        return True, flag
    elif (dealer_hand * vals).sum() < (player_hand * vals).sum():
        flag = 1
        return True, flag
    else:
        flag = 0
        return False, flag


def create_online_model(n_neurons: int = 32, verbose: bool=False, save_graph: bool=False) -> tf.keras.Model:
    """
    This function creates a neural network used for online training.
    :param n_neurons: integer. The number of neurons in the first layer of the model.
    :return: A compiled tensorflow.keras.Model object.
    """
    inputs = Input(shape=(22,))
    dense_1 = Dense(n_neurons, activation='elu',
                  kernel_initializer="he_uniform",
                  bias_initializer=initializers.RandomNormal()
                  )(inputs)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(n_neurons,
                  activation='elu',
                  kernel_initializer="he_uniform",
                  bias_initializer=initializers.RandomNormal()
                  )(dropout_1)
    dropout_2 = Dropout(0.5)(dense_2)
    outputs = Dense(4,
                  activation='softmax',
                  kernel_initializer="glorot_uniform",
                  )(dropout_2)
    model = Model(inputs=inputs, outputs=outputs, name="blackjack_model")
    model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    if verbose:
        model.summary()
    if save_graph:
        plot_model(model, "model_graph.png")
    return model


def run_simulation(online_model: tf.keras.Model, target_model: tf.keras.Model, deck: tuple):
    """
    This function runs a single instance of a game of 21.
    :param model: the "online" model that is trained on the outcome of each game.
    :param target_model: the target model is used to predict outcomes of a given hand, and its weights are updated
    :param deck: a tuple containing values representing the set of possible cards to draw from.
    :return: an integer value corresponding to the outcome of the simulation. Type A (0), B(1), or C(2).
    """
    vals = np.unique(deck)  # Card values
    y = np.array([[0] * 4])  # Container array for categorical outcome
    # Starting hands of cards for the player and dealer
    player_hand = create_card_vectors(2, deck)
    dealer_hand = create_card_vectors(1, deck)  # Dealer shows only one card at the start
    if player_hand[0][10] == 2:
        player_hand[0][10] -= 1
        player_hand[0][0] += 1
    if (player_hand * vals).sum() == 21:  # Instant win
        y[:, 0] += 1
        cards = np.concatenate((player_hand, dealer_hand), axis=1)
        online_model.fit(cards, y, verbose=False)
        return 0

    while True:
        
        # Array with the player hand and dealer hand represented as 1-D column vectors
        cards = np.concatenate((player_hand, dealer_hand), axis=1)

        # Use the target model to generate predictions based on the hand
        q_value_predictions = np.stack([target_model(cards, training=True) for sample in range(50)])
        q_values = q_value_predictions.mean(axis=0)[0]
        if q_values[0:3].sum() > q_values[3]:
            action = "stay"
        else:
            action = "hit"

        # Stay
        if action == "stay":
            outcome, flag = dealer(player_hand, dealer_hand, deck)  # Dealer routine returns outcome
            if outcome and flag:
                y[:, 0] += 1
                online_model.fit(cards, y, verbose=False)
                return 0
            elif outcome and not flag:
                y[:, 1] += 1
                online_model.fit(cards, y, verbose=False)
                return 1
            else:
                y[:, 3] += 1
                online_model.fit(cards, y, verbose=False)
                return 3

        # Hit
        if action == "hit":
            new_card = create_card_vectors(1, deck)  # Generate new card vector
            new_hand = player_hand + new_card  # Add new card to existing player hand
            while (new_hand * vals).sum() > 21:
                if new_hand[0][10] == 0:
                    y[:, 2] += 1
                    online_model.fit(cards, y, verbose=False)
                    return 2
                else:
                    new_hand[0][10] -= 1
                    new_hand[0][0] += 1
                    continue
            if (new_hand * vals).sum() <= 21:
                player_hand = new_hand  # Update 'player_hand' value (reiterate loop)
                continue


def run_simulations(n_train: int, n_update: int = 100, n_report: int = 500):
    """
    This function "manages" multiple instances of games of 21, and uses their outcomes to train reinforcement models
    by machine learning. Here, fixed Q-value targets (from 'target_model') are used to predict the outcome of a given
    hand, while the individual outcomes are used to train a separate "online" model. The weights from the "online"
    model (model) are used to update the target model at a user-specified rate (n_update).
    :param n_train: integer. The number of training iterations to run.
    :param n_update: integer. The number of iterations before the target model is updated from the online model.
    :param n_report: integer. The number of iterations before performance metrics are reported in the console window.
    :return: an integer value corresponding to the outcome of the simulation. Type A (0), B (1), or C (2).
    """
    online_model = create_online_model(32, verbose=False, save_graph=False)  #Create online model
    target_model = keras.models.clone_model(online_model)  # Create target model as copy of online model
    deck = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)  # Define card deck values
    outcomes = np.zeros(n_train)  # Container for training simulation outcomes
    for ite in range(n_train):  # Iterate over training loops
        outcomes[ite] = run_simulation(online_model=online_model, target_model=target_model, deck=deck)  # Save game outcome
        if ite % n_update == 0:
            target_model.set_weights(online_model.get_weights()) # Update the target model with weights from the online model
        if ite % n_report == 0 and ite > 0:
            # Calculate performance metrics and return results in the console
            unique, counts = np.unique(outcomes[:ite], return_counts=True)
            unique_run, counts_run = np.unique(outcomes[(ite - n_report):ite], return_counts=True)
            overall_win = np.round(counts[0:2].sum() / ite, 4)
            running_win = np.round(counts_run[0:2].sum() / n_report, 4)
            print(f"Iteration: {ite}/{n_train}...Overall Accuracy: {overall_win}...Running Accuracy: {running_win}")
    return online_model, target_model, outcomes


# Run game simulations (takes a bit of time)
n_train = 100000
online_model, target_model, outcomes = run_simulations(n_train=n_train, n_update=100, n_report=500)


# serialize model to JSON
model_json = online_model.to_json()
fname = "ai/models/online_model_3.json"
with open(fname, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
online_model.save_weights("ai/models/online_model_3.h5")
print("Saved model to disk")


# load json and compile model
json_file = open('ai/models/online_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("ai/models/online_model.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy")


result = []  # Save outcome of each round to container
n_iter = 5000  # Number of games to simulate
deck_of_cards = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)
for ite in range(0, n_iter):
    np.random.seed(222 * ite)  # For reproducability
    o = run_simulation(online_model, online_model, deck=deck_of_cards)
    result.append(o)
unique, counts = np.unique(result, return_counts=True)
total_wins = counts[0:2].sum()
print(f"The AI won {total_wins} out of {n_iter} rounds ({round(100*total_wins/n_iter,1)}%).")