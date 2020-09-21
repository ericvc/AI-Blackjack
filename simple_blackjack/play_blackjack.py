import os
import sys as sys
from simple_blackjack import *
import numpy as np


# get size of terminal window for positioning print text
width = os.get_terminal_size().columns


opt_ai = input("Would you like to use the AI assistant? (Yes/No): ".center(width))
#Initialize player's hand
start_amount = get_starting_amount() # prompts user
player = Player(start_amount)
if opt_ai.lower() == "yes":
    player.load_model()


print("........................................\n".center(width))
print("Starting new game...".center(width))


def main():

    print("........................................\n".center(width))

    # enter bet for a single hand
    bet = player.enter_bet()
    # initialize player cards for the round
    player.cards = [] #  reset player hand
    player.cards += [deal_cards(2)]
    player.n_rounds += 1

    # option to split the deck if starting cards are the same
    if player.cards[0][0] == player.cards[0][1]:
        print(f"Your starting cards for this hand are [{player.cards[0][0]}] and [{player.cards[0][1]}].".center(width))
        opt_split = input("Would you like to split the hand? (Yes/No): ")
        if opt_split.lower()[0] == "y":
            player.n_rounds += 1
            player.cards = [[player.cards[0][0]] + deal_cards(1),
                            [player.cards[0][1]] + deal_cards(1)]
	    dealer_cards = [deal_cards(1)]

    for game in range(player.n_rounds):

        # initialize dealer hand
        dealer = Dealer()
	if dealer_cards:
	    dealer.cards += dealer_cards
	else:
	    dealer.cards += [deal_cards(1)]

        # check for instant win
        if player.total(game) == 21:
            print("Blackjack! You won the hand.\n".center(width))
            player.win(bet)
            continue

        print(f"Your starting cards for this hand are [{player.cards[game][0]}] and [{player.cards[game][1]}]."
              f" The dealer is showing {dealer.cards[0]}\n".center(width))

        # check if starting cards are a bust (2 x Ace)
        if player.total(game) == 22:
            print("Converting ace to value of 1.\n".center(width))
            player.cards[game][0] = 1

        while player.total(game) <= 21:

            if opt_ai.lower() == "yes":
                player.predict_outcome(player.cards[game], dealer.cards[0])
            
            opt = input("What would like to do? [1 - Hit | 2 - Stay | 3 - Exit]: ")
            print("\n")

            if opt.lower() == "hit" or opt.lower() == "1":

                new_card = deal_cards(1)
                player.cards[game] += new_card
                print(f"You draw {new_card}.".center(width))
                print(f"Your card total is {player.total(game)}\n".center(width))

                if player.total(game) > 21 and 11 in player.cards[game]:
                    ace_position = int(np.where(np.array(player.cards[game]) == 11)[0][0])
                    player.cards[game][ace_position] = 1
                    print("Converting ace to value of 1.\n".center(width))
                    print(f"Your cards are {player.cards[game]}.".center(width))
                    print(f"Your card total is {player.total(game)}\n".center(width))

                if player.total(game) > 21 and not 11 in player.cards[game]:
                    print("Bust.\n".center(width))
                    player.lose(bet)

            elif opt.lower() == "stay" or opt.lower() == "2":

                print("*You choose to stay*".center(width))
                print(f"Dealer's hand is {dealer.cards[0]}\n".center(width))

                while dealer.total() < player.total(game) and dealer.total() < 17:
                    new_card = deal_cards(1)
                    dealer.cards[0] += new_card
                    print(f"Dealer draws {new_card}.\n".center(width))

                    if dealer.total() > 21 and 11 in dealer.cards[0]:
                        ace_position = int(np.where(np.array(dealer.cards) == 11)[0][0])
                        dealer.cards[0][ace_position] = 1
                        print("Converting ace to value of 1.\n".center(width))
                        print(f"Dealer's cards are {dealer.cards[0]}.".center(width))
                        print(f"Dealer's card total is {dealer.total()}\n".center(width))
                        continue

                if player.total(game) <= dealer.total() <= 21:
                    print(f"Dealer's card total is {dealer.total()}\n".center(width))
                    print(f"Dealer wins [{dealer.total()}]\n".center(width))
                    player.lose(bet)
                    break

                if dealer.total() > 21:
                    print(f"Dealer went bust [{dealer.total()}].\n".center(width))
                    player.win(bet)
                    break
                    
                if player.total(game) > dealer.total() >= 17:
                    print(f"Dealer's card total is {dealer.total()}\n".center(width))
                    player.win(bet)
                    break

            elif opt.lower() == "exit" or opt.lower() == "3":
                print(f"Thanks for playing. Session status: You won {player.total_wins} out of {player.total_games} rounds\n".center(width))
                sys.exit(0)

    if player.money <= 0:
        print("You are out of money. Goodbye!".center(width))
        sys.exit(0)

while True:
    main()
