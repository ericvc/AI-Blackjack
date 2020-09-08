import os
import random
import sys as sys
from sys import argv
# from simple_blackjack import Player
# from simple_blackjack import Dealer
# from simple_blackjack import deal_cards
# from simple_blackjack import enter_bet
from simple_blackjack import *
import numpy as np


# gets size of terminal window for positioning text
width = os.get_terminal_size().columns


print("........................................\n".center(width))
print("Starting new game...".center(width))


#Initialize player's hand
player = Player(get_starting_amount())


def main():

    print("........................................\n".center(width))

    opt0 = input("What would like to do? (Play/Exit): ")
    if opt0.lower() == "exit":
        sys.exit(0)

    # enter bet for a single hand
    bet = enter_bet(player)
    # initialize player cards for a single hand
    player.cards = [] #reset player hand
    player.cards += [deal_cards(2)]
    
    #initialize dealer hand
    dealer = Dealer()

    # option to split the deck if starting cards are equal
    if player.cards[0][0] == player.cards[0][1]:
        print(f"Your starting cards for this hand are {player.cards[0][0]} and {player.cards[0][1]}. The dealer is showing a [{dealer.cards[0]}]\n".center(width))
        opt_split = input("Would you like to split the hand? (Yes/No): ".center(width))
        if opt_split.lower() == "yes":
            player.cards = [[player.cards[0][0]] + deal_cards(1),
                            [player.cards[0][1]] + deal_cards(1)]

    for player_cards in player.cards:

        print(f"Your starting cards for this hand are {player.cards[0][0]} and {player.cards[0][1]}. The dealer is showing [{dealer.cards[0]}]\n".center(width))

        # check for instant win
        if sum(player.cards[0]) == 21:
            player.money += bet
            print("Blackjack! You won the hand.\n".center(width))
            print(f"You won ${bet:.2f} (Total: ${player.money:.2f})\n".center(width))
            break

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

                while sum(dealer.cards) < sum(player_cards) and sum(dealer.cards) < 17:
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
                    
                if sum(dealer.cards) < sum(player_cards) and sum(dealer.cards) >= 17:
                    player.money += bet
                    print(f"Dealer's card total is {sum(dealer.cards)}\n".center(width))
                    print(f"You won ${bet:.2f} (Total: ${player.money:.2f})\n".center(width))
                    break

        if opt0.lower() == "exit":
            sys.exit(0)

    if player.money <= 0:
        print("You are out of money. Goodbye!".center(width))
        sys.exit(0)

while True:
    main()
