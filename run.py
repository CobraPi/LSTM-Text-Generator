from TwitterBot import TwitterBot


def print_menu():
    print("Welcome to the Twitter Tweet Generator")
    print("Options:")
    print("Enter 0 to exit")
    print("Enter 1 to generate text with a random seed")
    print("Enter 2 to generate text with a supplied seed")


def run_model(name):
    bot = TwitterBot()
    bot.read_corpus_file("data/" + name + ".txt")
    #bot.set_outpufile("generated_text/" + name + ".txt")
    bot.load_saved_model("models/LSTM_MODEL-epoch100-words60958-sequence10-minfreq20-loss0.7329-acc0.9189-val_loss13.2298-val_acc0.0319")
    user_input = 1
    while int(user_input) != 0:
        print_menu()
        user_input = input(">>> ")
        if int(user_input) == 1:
            bot.generate_text_on_run()
        elif int(user_input) == 2:
            seed_verified = False
            while not seed_verified:
                seed = input("Please enter a seed sentence: ")
                seed_verified = bot.seed_in_vocabulary(seed)
                if not seed_verified:
                    print("Not all words in seed are in vocabulary. Please try again.")
            bot.generate_text_on_run(seed, True)
        elif int(user_input) == 0:
            break
        else:
            print("Invalid input, please try again")

if __name__ == "__main__":
    run_model("cambrasine_tweets")
