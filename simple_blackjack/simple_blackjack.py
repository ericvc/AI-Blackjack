import os
from sys import argv
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # quiet loading TensorFlow


# get size of terminal window for positioning print text
width = os.get_terminal_size().columns


class Dealer:
    def __init__(self):
        self.cards = [] #  hand of cards
    
    def total(self):
        return sum(self.cards[0])
        

class Player:
    def __init__(self, money: float):
        self.money = money #  total balance
        self.cards = [] #  hand of cards
        self.n_rounds = 0 #  number of rounds remaining (usually 1)
        self.total_wins = 0 #  total number of wins per session
        self.total_games = 0 #  total number of games played in a session

    def load_model(self):
        # load json and compile model
        json_file = open('ai/models/online_model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        trained_model = model_from_json(model_json)
        # load weights into new model
        trained_model.load_weights("ai/models/online_model.h5")
        trained_model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy")
        self.trained_model = trained_model

    def predict_outcome(self, player_cards: list, dealer_cards: list):
        cards = self.cards_to_card_vectors(player_cards, dealer_cards)
        q_value_samples = np.stack([self.trained_model(cards, training=True) for sample in range(50)])
        q_values = q_value_samples.mean(axis=0)[0]
        if q_values[0:3].sum() > q_values[3]:
            action = f"**Stay (%.2f)**" % (q_values[0:3].sum())
        else:
            action = "**Hit (%.2f)**" % (q_values[3])
        print(f"AI Recommends: {action}")

    def cards_to_card_vectors(self, player_cards, dealer_cards):
        n_cards = 11
        p_cardvec = np.array([[0] * n_cards])
        d_cardvec = np.array([[0] * n_cards])
        for card in player_cards:
            p_cardvec[:, card - 1] += 1
        for card in dealer_cards:
            d_cardvec[:, card - 1] += 1
        cards = np.concatenate((p_cardvec, d_cardvec), axis=1)
        return cards

    def win(self, bet):
        self.money += bet
        self.total_games += 1
        self.total_wins += 1
        self.n_rounds -= 1
        print(f"You won ${bet:.2f} (Total: ${self.money:.2f})\n".center(width))

    def lose(self, bet):
        self.money -= bet
        self.total_games += 1
        self.n_rounds -= 1
        print(f"You lost ${bet:.2f} (Total: ${self.money:.2f})\n".center(width))
    
    def total(self, game):
        return sum(self.cards[game])

    def enter_bet(self):
        """
        This function prompts the user to enter a bet amount. It then checks that this input is valid.
        :return: The verified bet amount is returned after ensuring it meets the specified criteria.
        """
        bet_amount = ""
        while not isinstance(bet_amount, float) or bet_amount > self.money:
            try:
                bet_amount = round(float(input(f"How much would you like to wager (balance: ${player.money:.2f})? ")), 2)
                if bet_amount > self.money:
                    print("You don't have enough money.\n")
                    continue
            except ValueError:
                print("Invalid selection! Try again.")
                continue
        return (bet_amount)
    

def deal_cards(n: int):
    """
    This function randomly samples `n` cards from a "deck of cards" (with replacement)
    :param n: integer number of cards to draw from the deck
    :return: n cards sampled from the deck.
    """
    # 'deck' represents the possible card values to be dealt to the player
    deck = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)
    return list(np.random.choice(deck, n, replace=True))


def get_starting_amount():
    """
    This function check that the starting player balance meets criteria: a flpating point value greater than zero.
    :return: A floating point representing the initial player balance.
    """
    if len(argv) == 1:  # default starting amount
        start_amount = 100.0
    elif len(argv) == 2:  # user-specified starting amount
        start_amount = float(argv[1])
    while not isinstance(start_amount, float) or start_amount <= 0:
        try:
            start_amount = round(float(input("You must enter a positive amount of money to begin: ")), 2)
        except ValueError:
            print("Invalid selection! Try again.")
            continue
    return start_amount
