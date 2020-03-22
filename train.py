from TwitterBot import TwitterBot
from datetime import datetime


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
    bot.save_model("models/Cambrasine_double_layer_double_trained.h5")

if __name__ == "__main__":
    #train_new_cambrasine_model()
    train_existing_cambrasine_model("models/Cambrasine_double_layer.h5")
