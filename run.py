from TextGenerator import TextGenerator
from datetime import datetime

class Model:
    def __init__(self, name, corpus_filepath, model_filepath, num_layers, embedding=True):
        self.name = name
        self.corpus_filepath = corpus_filepath
        self.model_filepath = model_filepath
        self.num_layers = num_layers
        self.embedding = embedding
        self.model = TextGenerator(name, embedding, num_layers)

    def load(self):
        self.model.load_saved_model(self.model_filepath)
        self.model.read_corpus_file(self.corpus_filepath)

    def generate_text(self):
        print("Generating text for:", self.name)
        self.model.generate_text_on_run()


def run_model(alicepath, blakepath, trumppath):
    alice = TextGenerator()
    alice.load_saved_model(alicepath)
    alice.read_corpus_file("data/alice.txt")
    blake = TextGenerator()
    blake.load_saved_model(blakepath)
    blake.read_corpus_file("data/blake_poems.txt")
    trump = TextGenerator()
    trump.load_saved_model(trumppath)
    trump.read_corpus_file("data/trump_tweets.txt")
    state = True
    while state:
        print("General Text Generator")
        print("Enter a for Alice in Wornderland")
        print("Enter: b for William blakes poems")
        print("Enter: t for Trumps tweets")
        print("Enter e to exit")
        user = input("Choice: ")
        if user == "a":
            print("Alice In Wonderland")
            alice.generate_text_on_run()
        elif user == "b":
            print("William Blake's Poems")
            blake.generate_text_on_run()
        elif user == "t":
            print("Trump's tweets")
            trump.generate_text_on_run()
        elif user == "e":
            state = False
        else:
            print("Not a valid command")

trump_info = {"name": "Trump", "corpus_filepath": "data/trump_tweets.txt", "model_filepath": "models/trump_double_layer_embedding.h5", "num_layers": 2, "embedding": True }

if __name__ == "__main__":
    #run_model("models/alice_double_layer_embedding.h5", "models/blake_double_layer_embedding.h5", "models/trump_and_cambrasine_five_layer_lstm.h5")
    models = []
    models.append(trump_info)
    for item in models:
        print(item)
        #tmp_model.load()
        #models[item["name"]] = tmp_model

    state = True
    while state:
        print("General Text Generator")
        print("Enter a for Alice in Wornderland")
        print("Enter: b for William blakes poems")
        print("Enter: t for Trumps tweets")
        print("Enter e to exit")
        user_choice = input("Choice: ")
        if user_choice == "t":
            tmp_model = models["Trump"]
            tmp_model.generate_text()
        elif user_choice == "e":
            state = False
