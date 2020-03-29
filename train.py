from TwitterBot import TwitterBot
from datetime import datetime


def train_new_trump_model():
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=3)
    bot.read_corpus_file("data/trump_tweets.txt")
    bot.set_outpufile("generated_text/generated_trump_tweets.txt" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/trump_double_layer_embedding.h5")


def train_existing_cambraine_trump(filepath):
    current_time = datetime.now()
    bot = TwitterBot(embedding=True, model_layers=2)
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.load_saved_model(filepath)
    print("Training model:", filepath)
    bot.train()
    bot.save_model("models/Cambrasine_double_layer_embedding.h5")

if __name__ == "__main__":
    train_new_trump_model()
    #train_existing_cambrasine_model("models/LSTM_MODEL_EMBEDDING_2_LAYERS-epoch009-words62992-sequence50-minfreq20-loss7.2008-acc0.1281-val_loss7.7315-val_acc0.1149")
