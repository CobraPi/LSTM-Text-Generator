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


def train_new_cambrasine_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/cambrasine_double_layer_embedding.h5")


def train_new_alice_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/alice.txt")
    bot.set_outpufile("generated_text/generated_alice_text" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/alice_double_layer_embedding.h5")


def train_new_bible_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/bible.txt")
    bot.set_outpufile("generated_text/generated_bible_text(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/bible_double_layer_embedding.h5")


def train_new_blake_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/blake_poems.txt")
    bot.set_outpufile("generated_text/generated_blake_text(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/blake_double_layer_embedding.h5")


def train_new_odyssey_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/odyssey.txt")
    bot.set_outpufile("generated_text/generated_odyssey_text(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/odyssey_double_layer_embedding.h5")


if __name__ == "__main__":
    print("Training blake") 
    train_new_blake_model()
    print("Training trump") 
    train_new_trump_model()
    print("Training alice") 
    train_new_alice_model()
    print("Training bible") 
    train_new_bible_model()
    print("Training odyssey") 
    train_new_odyssey_model()
    print("Training cambrasine") 
    train_new_cambrasine_model()
