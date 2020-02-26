from TwitterBot import TwitterBot

if __name__ == "__main__":

    bot = TwitterBot()
    bot.read_corpus("data/cambrasine_tweets.txt")
    bot.set_outpufile("data/generated_text.txt")
    bot.build_model(1)
    bot.train()
    bot.save_model("Cambrasine_single_layer.h5")

