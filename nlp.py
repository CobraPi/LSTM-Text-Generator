import csv
import nltk
import matplotlib
import pandas
from itertools import chain

def plot_freq_dist(filepath):
    tokens = []
    tokenizer = nltk.RegexpTokenizer(r"\b\w+\S+")
    with open(filepath, "r") as csv_in:
        reader = csv.reader(csv_in, delimiter=",")

        for row in reader:
            rt = tokenizer.tokenize(row[3])

            for word in rt:
                tokens.append(word)

    freq_dist = nltk.FreqDist(tokens)
    freq_dist.plot(50, cumulative=False)


if __name__ == "__main__":

    filepath = "data/cambrasine_only_tweets.csv"
    df = pandas.read_csv(filepath)
    tokenizer = nltk.RegexpTokenizer(r"\b\w+\S+")

    sents = df["text"].map(tokenizer.tokenize).tolist()

    freq_dist = nltk.FreqDist(list(chain(*[nltk.ngrams(sent, 7) for sent in sents])))

    freq_dist.plot(30, cumulative=False)
