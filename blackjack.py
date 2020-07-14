import os
import random
import sys as sys
from sys import argv

import numpy as np


class Dealer:
    def __init__(self):
        self.cards = deal_cards(2)


class Player:
    def __init__(self, money):
        self.money = float(money)
        self.cards = None


# Define function for accepting user input (guess) and makem sure it is an integer between 1 and 100
def enter_bet(player):
    bet_amount = ""
    while not isinstance(bet_amount, float) or bet_amount > player.money:
        try:
            bet_amount = round(float(input(f"How much would you like to wager (balance: ${player.money:.2f})? ")), 2)
            if bet_amount > player.money:
                print("You don't have enough money.\n")
                continue
        except ValueError:
            print("Invalid selection! Try again.")
            continue
    return (bet_amount)


# Define function for accepting user input (guess) and making sure it is an integer between 1 and 100
def get_starting_amount():
    # Check that user-specificed starting amount is greater than zero
    if len(argv) == 1:  # default starting amount
        start_amount = float(100)
    elif len(argv) == 2:  # user-specified starting amount
        start_amount = float(argv[1])
    while not isinstance(start_amount, float) or start_amount <= 0:
        try:
            start_amount = round(float(input("You must enter a positive amount of money to begin: ")), 2)
        except ValueError:
            print("Invalid selection! Try again.")
            continue
    return start_amount


def deal_cards(n: int):
    # 'deck' represents the possible card values to be dealt to the player
    deck = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11)
    return random.sample(deck, n)


# gets size of terminal window for positioning text
width = os.get_terminal_size().columns

print("........................................\n".center(width))
print("Starting new game...".center(width))

player = Player(get_starting_amount())


def main():
    player.cards = []

    print("........................................\n".center(width))

    opt0 = input("What would like to do? (Play/Exit): ")
    if opt0.lower() == "exit":
        sys.exit(0)

    # enter bet for a single hand
    bet = enter_bet(player)
    # initialize player cards for a single hand
    player.cards += [deal_cards(2)]
    # print starting cards
    print(f"Your starting cards are {player.cards[0][0]} and {player.cards[0][1]}\n.".center(width))

    # check for instant win
    if sum(player.cards[0]) == 21:
        player.money += bet
        print("Blackjack! You won the hand.\n".center(width))
        print(f"You won ${bet:.2f} (Total: ${player.money:.2f})\n".center(width))

    # option to split the deck if starting cards are equal
    if player.cards[0][0] == player.cards[0][1]:
        opt_split = input("Would you like to split the hand? (Yes/No): ")
        if opt_split.lower() == "yes":
            player.cards = [[player.cards[0][0]] + deal_cards(1),
                            [player.cards[0][1]] + deal_cards(1)]

    for player_cards in player.cards:

        # initialize dealer's hand
        dealer = Dealer()

        print(f"Your starting cards for this hand are {player_cards[0]} and {player_cards[1]}\n.".center(width))

        # check if starting cards are a bust (2 x Ace)
        if sum(player_cards) == 22:
            print("Converting ace to value of 1.\n".center(width))
            player_cards[0] = 1

        while sum(player_cards) < 21:

            opt = input("What would like to do? (Hit/Stay): ")
            if opt.lower() == "hit":

                new_card = deal_cards(1)
                player_cards += new_card
                print(f"You draw {new_card}.".center(width))
                print(f"Your card total is {sum(player_cards)}\n".center(width))

                if sum(player_cards) == 21:
                    print("Your total is 21.\n".center(width))
                    opt = "stay"

                if sum(player_cards) > 21 and 11 in player_cards:
                    ace_position = int(np.where(np.array(player_cards) == 11)[0][0])
                    player_cards[ace_position] = 1
                    print("Converting ace to value of 1.\n".center(width))
                    print(f"Your cards are {player_cards}.".center(width))
                    print(f"Your card total is {sum(player_cards)}\n".center(width))

                if sum(player_cards) > 21 and not 11 in player_cards:
                    player.money -= bet
                    print("Bust.\n".center(width))
                    print(f"You lost ${bet:.2f} (Total: ${player.money:.2f})\n".center(width))

            if opt.lower() == "stay":

                print("*You choose to stay*".center(width))
                print(f"Dealers hand is {dealer.cards}\n".center(width))

                while sum(dealer.cards) < sum(player_cards):
                    new_card = deal_cards(1)
                    dealer.cards += new_card
                    print(f"Dealer draws {new_card}.\n".center(width))

                    if sum(dealer.cards) > 21 and 11 in dealer.cards:
                        ace_position = int(np.where(np.array(dealer.cards) == 11)[0][0])
                        dealer.cards[ace_position] = 1
                        print("Converting ace to value of 1.\n".center(width))
                        print(f"Dealer's cards are {dealer.cards}.".center(width))
                        print(f"Dealer's card total is {sum(dealer.cards)}\n".center(width))
                        continue

                if sum(dealer.cards) >= sum(player_cards) and sum(dealer.cards) <= 21:
                    player.money -= bet
                    print(f"Dealer's card total is {sum(dealer.cards)}\n".center(width))
                    print(f"Dealer wins [{sum(dealer.cards)}]\n".center(width))
                    print(f"You lost ${bet:.2f} (Total: ${player.money:.2f})\n".center(width))
                    break

                if sum(dealer.cards) > 21 and not 11 in dealer.cards:
                    player.money += bet
                    # print(f"Dealer's card total is {sum(dealer.cards)}\n".center(width))
                    print(f"Dealer went bust [{sum(dealer.cards)}].\n".center(width))
                    print(f"You won ${bet:.2f} (Total: ${player.money:.2f})\n".center(width))
                    break

        if opt0.lower() == "exit":
            sys.exit(0)

    if player.money <= 0:
        print("You are out of money. Goodbye!".center(width))
        sys.exit(0)

while True:
    main()
