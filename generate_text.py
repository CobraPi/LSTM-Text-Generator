import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load tweets as a huge string and inserts periods at the end of
# a tweet if it doesn't have one
raw_text = ""
filepath = "data/cambrasine_only_tweets.csv"
with open(filepath, "r") as csvin:
    reader = csv.reader(csvin, delimiter=",")
    for row in reader:
        if not row[3].endswith("."):
            raw_text += row[3] + ". "
        else:
            raw_text += row[3] + " "

# create set of distinct characters and map each one to a unique integer
chars = sorted(list(set(raw_text)))
char_map = dict((c, i) for i, c in enumerate(chars))

# summarize dataset
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters:", n_chars)
print("Vocabulary Size:", n_vocab)

# create training data by encoding the input to output pairs as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_map[char] for char in seq_in])
    dataY.append(char_map[seq_out])
n_patterns = len(dataX)
print("Total Patterns:", n_patterns)

# transform the list of input sequences into the form [samples, time steps, features]
x = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
x = x / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
# load the network weights
filename = "" # weights-iomporvement file with lowest loss value
model.load_weights(filename)
model.compile(loss="categorical_crossentrophy", optimizer="adam")

# create mapping of unique characters to integers
int_to_char = dict((i, c) for i, c in enumerate(chars))
