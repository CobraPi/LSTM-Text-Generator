from TwitterBot import TwitterBot
from datetime import datetime
#from nltk.corpus import gutenberg as gt

"""
def create_cambrasine_model():
    current_time = datetime.now()
    bot = TwitterBot()
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_text(" + str(datetime) + ").txt")
    # bot.load_saved_model("checkpoints/LSTM_LYRICS-epoch003-words70742-sequence10-minfreq10-loss7.8825-acc0.1058-val_loss8.0334-val_acc0.1058")
    # bot.generate_text()
    bot.build_model(2)
    bot.train()
    bot.save_model("models/Cambrasine_double_layer.h5")
"""
"""
def create_bible_model():
    current_time = datetime.now()
    raw_text = gt.raw("bible-kjv.txt")
    bot = TwitterBot()
    bot.read_corpus(raw_text)
    bot.set_outpufile("generated_text/generated_bible(" + str(datetime) + ").txt")
    bot.build_model(2)
    bot.train()
    bot.save_model("models/bible_double_layer.h5")
"""


def train_new_cambrasine_model():
    current_time = datetime.now()
    bot = TwitterBot()
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.build_model(2)
    bot.train()
    bot.save_model("models/Cambrasine_double_layer.h5")


def train_existing_cambrasine_model(filepath):
    current_time = datetime.now()
    bot = TwitterBot()
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.load_saved_model(filepath)
    print("Training model:", filepath)
    bot.train()
    bot.save_model("models/Cambrasine_double_layer.h5")


def print_menu():
    print("Welcome to the Twitter Tweet Generator")
    print("Options:")
    print("Enter 0 to exit")
    print("Enter 1 to generate text with a random seed")
    print("Enter 2 to generate text with a supplied seed")


def run_cambrasine():
    bot = TwitterBot()
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("cambrasine_output.txt")
    #bot.load_saved_model("models/LSTM_LYRICS-epoch003-words70742-sequence10-minfreq10-loss7.8825-acc0.1058-val_loss8.0334-val_acc0.1058")
    bot.build_model(2)
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


if __name__ == "__main__":
    train_new_cambrasine_model()
    #train_existing_cambrasine_model("models/Cambrasine_double_layer.h5")
