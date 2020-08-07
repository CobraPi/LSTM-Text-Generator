from TwitterBot import TwitterBot
from datetime import datetime


def run_model(alicepath, blakepath):
    alice = TwitterBot()
    alice.load_saved_model(alicepath)
    alice.read_corpus_file("data/alice.txt")
    blake = TwitterBot()
    blake.load_saved_model(blakepath)
    blake.read_corpus_file("data/blake_poems.txt")
    state = True
    while state:
        print("General Text Generator")
        print("Enter a for Alice in Wornderland")
        print("Enter: b for William blakes poems")
        print("Enter e to exit")
        user = input("Choice: ")
        if user == "a":
            print("Alice In Wonderland")
            alice.generate_text_on_run()
        elif user == "b":
            print("William Blake's Poems")
            blake.generate_text_on_run()
        elif user == "e":
            state = False
        else:
            print("Not a valid command")


if __name__ == "__main__":
    run_model("models/alice_double_layer_embedding.h5", "models/blake_double_layer_embedding.h5")





