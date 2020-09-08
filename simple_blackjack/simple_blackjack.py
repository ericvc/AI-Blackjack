import os
import random
import sys as sys
from sys import argv
import numpy as np


def deal_cards(n: int):
  """
  This function randomly samples `n` cards from a "deck of cards" (with replacement)
  :param n: integer number of cards to draw from the deck
  :return: n cards sampled from the deck.
  """
  # 'deck' represents the possible card values to be dealt to the player
  deck = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)
  return random.sample(deck, n)


class Dealer:
    def __init__(self):
        self.cards = deal_cards(2)
        

class Player:
    def __init__(self, money: float):
        self.money = money
        self.cards = []
        
        
def enter_bet(player):
    """
    This function accepts 'Player' class objects as inputs and prompts the user to enter a bet amount. It then
    checks that this input is valid by ensuring it is a floating point value less than or equal to the player's
    remaining balance.
    :param player: 'Player' class object.
    :return: The verified bet amount is returned after ensuring it meets the specified criteria.
    """
    bet_amount = ""
    while not isinstance(bet_amount, float) or bet_amount > player.money:
        try:
            bet_amount = round(float(input(f"How much would you like to wager (balance: ${player.money:.2f})?")), 2)
            if bet_amount > player.money:
                print("You don't have enough money.\n")
                continue
        except ValueError:
            print("Invalid selection! Try again.")
            continue
    return (bet_amount)
    

def get_starting_amount():
    """
    This function check that the starting player balance meets criteria: a floating point value greater than zero.
    :return: A floating point representing the initial player balance.
    """
    if len(argv) == 1:  # default starting amount
        start_amount = 100
    elif len(argv) == 2:  # user-specified starting amount
        start_amount = float(argv[1])
    while not isinstance(start_amount, float) or start_amount <= 0:
        try:
            start_amount = round(float(input("You must enter a positive amount of money to begin: ")), 2)
        except ValueError:
            print("Invalid selection! Try again.")
            continue
    return start_amount
    

