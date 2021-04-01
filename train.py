from TextGenerator import TextGenerator
from datetime import datetime


def train_new_trump_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="TRUMP", embedding=True, epochs=50, model_layers=2)
    bot.read_corpus_file("data/trump_tweets.txt")
    bot.set_outpufile("generated_text/generated_trump_tweets.txt" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/trump_two_layer_lstm.h5")


def train_new_cambrasine_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="CAMBRASINE", embedding=True, model_layers=2)
    bot.read_corpus_file("data/cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_cambrasine_tweets(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/cambrasine_double_layer_embedding.h5")


def train_trump_and_cambrasine_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="TRUMP_CAMBRASINE", embedding=True, epochs=10, model_layers=7)
    bot.read_corpus_file("data/trump_and_cambrasine_tweets.txt")
    bot.set_outpufile("generated_text/generated_trump_and_cambrasine_tweets.txt" + str(current_time) + ").txt")
    #bot.get_model()
    bot.load_saved_model("checkpoints/TRUMP_CAMBRASINE_LSTM_MODEL_EMBEDDING_5_LAYERS-epoch011-words131472-sequence20-minfreq20-loss6.4050-val_loss6.8033-acc0.1501")
    bot.train()
    bot.save_model("models/trump_and_cambrasine_five_layer_lstm.h5")


def train_new_alice_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="ALICE", embedding=True, epochs=1000, model_layers=3)
    bot.read_corpus_file("data/alice.txt")
    bot.set_outpufile("generated_text/generated_alice_text" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/alice_five_layer_gru.h5")


def train_new_bible_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="BIBLE", embedding=True, model_layers=2)
    bot.read_corpus_file("data/bible.txt")
    bot.set_outpufile("generated_text/generated_bible_text(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/bible_double_layer_embedding.h5")


def train_new_blake_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="BLAKE", embedding=True, model_layers=2)
    bot.read_corpus_file("data/blake_poems.txt")
    bot.set_outpufile("generated_text/generated_blake_text(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/blake_double_layer_embedding.h5")


def train_new_odyssey_model():
    current_time = datetime.now()
    bot = TextGenerator(checkpoint_name="ODYSSEY", embedding=True, model_layers=2)
    bot.read_corpus_file("data/odyssey.txt")
    bot.set_outpufile("generated_text/generated_odyssey_text(" + str(current_time) + ").txt")
    bot.get_model()
    bot.train()
    bot.save_model("models/odyssey_double_layer_embedding.h5")


def train_test_model():
    bot = TextGenerator(checkpoint_name="TEST", embedding=True, model_layers=2)
    bot.read_corpus_file("data/trump_tweet_test.txt")
    bot.load_w2v_model()
    #bot.train_w2v_model()
    bot.get_model()
    bot.train()





if __name__ == "__main__":
    print("Training trump and cambrasine")
    #train_trump_and_cambrasine_model()
    #train_test_model()
    #print("Training blake")
    #train_new_blake_model()
    #print("Training alice")
    #train_new_alice_model()
    print("Training trump")
    train_new_trump_model()
    #print("Training bible")
    #train_new_bible_model()
    #print("Training odyssey")
    #train_new_odyssey_model()
    #print("Training cambrasine")
    #train_new_cambrasine_model()
