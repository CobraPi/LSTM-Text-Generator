from TwitterBot import TwitterBot
from datetime import datetime


def train_new_trump_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/trump_tweets.txt")
    bot.set_outpufile("generated_text/generated_trump_tweets.txt" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/trump_double_layer_embedding.h5")

def train_new_alice_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/alice.txt")
    bot.set_outpufile("generated_text/generated_alice" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/alice_double_layer_non-embedding.h5")

def train_new_cambrasine_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.get_model() 
    bot.train()
    bot.save_model("models/Cambrasine_double_layer_embedding.h5")


def train_existing_cambraine_model(filepath):
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.load_saved_model(filepath)
    print("Training model:", filepath)
    bot.train()
    bot.save_model("models/Cambrasine_double_layer_embedding.h5")


if __name__ == "__main__":
    #train_new_alice_model()
    train_new_trump_model()
    #train_new_cambrasine_model()
