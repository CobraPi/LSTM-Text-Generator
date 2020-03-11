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
    bot.set_outpufile("generated_text/" + name + ".txt")
    bot.load_saved_model("models/LSTM_MODEL-epoch047-words60957-sequence10-minfreq20-loss0.8391-acc0.8813-val_loss12.8418-val_acc0.0355")
    user_input = 1
    while int(user_input) != 0:
        print_menu()
        user_input = input(">>> ")
        if int(user_input) == 1:
            bot.generate_text()
        elif int(user_input) == 2:
            seed = input("Please enter a seed sentence: ")
            bot.generate_text(seed, True)
        elif int(user_input) == 0:
            break
        else:
            print("Invalid input, please try again")
    bot.outputfile.close()

if __name__ == "__main__":
    run_model("cambrasine_tweets")
