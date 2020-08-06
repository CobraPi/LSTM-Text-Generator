from TwitterBot import TwitterBot
from datetime import datetime


def run_model(filepath):
    bot = TwitterBot()
    bot.load_saved_model(filepath)
    bot.read_corpus_file("data/blake_poems.txt")
    state = True
    while True:
        input("Enter")
        seed = input("Enter Seed (1 to exit): ")
        bot.generate_text_on_run(seed)


if __name__ == "__main__":
    run_model("models/blake_double_layer_embedding.h5")





