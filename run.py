from TwitterBot import TwitterBot


def print_menu():
    print("Welcome to the Twitter Tweet Generator")
    print("Options:")
    print("Enter 0 to exit")
    print("Enter 1 to generate text with a random seed")
    print("Enter 2 to generate text with a supplied seed")


def run_model(corpora, model):
    bot = TwitterBot(embedding=True)
    bot.read_corpus_file("data/" + corpora + ".txt")
    #bot.set_outpufile("generated_text/" + name + ".txt")
    bot.load_saved_model(model)
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
    run_model("cambrasine", "models/LSTM_MODEL_EMBEDDING-epoch100-words2415-sequence10-minfreq20-loss0.8112-acc0.7574-val_loss8.8984-val_acc0.1000")
