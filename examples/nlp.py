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

    filepath = "../data/cambrasine_only_tweets.csv"

    average = []

    with open(filepath, "r") as csvin:
        reader = csv.reader(csvin, delimiter=",")

        for row in reader:
            average.append(len(row[3]))

    print(str(sum(average) / len(average)))

    df = pandas.read_csv(filepath)

    tokenizer = nltk.RegexpTokenizer(r"\b\w+\S+")

    sents = df["text"].map(tokenizer.tokenize).tolist()
    print(sents)
    freq_dist = nltk.FreqDist(list(chain(*[nltk.ngrams(sent, 7) for sent in sents])))

    freq_dist.plot(30, cumulative=False)
