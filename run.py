from TwitterBot import TwitterBot
from datetime import datetime


def run_alice_model(filepath):
    bot = TwitterBot()
    bot.load_saved_model(filepath)
    bot.read_corpus_file("data/alice.txt")
    state = True
    while True:
        input("Enter")
        seed = input("Enter Seed (1 to exit): ")
        bot.generate_text_on_run(seed, True)


if __name__ == "__main__":
    run_alice_model("checkpoints/LSTM_MODEL_EMBEDDING_2_LAYERS-epoch095-words6457-sequence20-minfreq20-loss0.0295-val_loss20.2823-accuracy0.9910")





