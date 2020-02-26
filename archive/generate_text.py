import numpy #as np
import sys
import csv
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
filepath = "../data/cambrasine_only_tweets.csv"
with open(filepath, "r") as csvin:
    reader = csv.reader(csvin, delimiter=",")
    for row in reader:
        if not row[3].endswith("."):
            raw_text += row[3] + ". "
        else:
            raw_text += row[3] + " "

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "../weights-improvement-06-2.4779.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
sys.stdout.flush()
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
	sys.stdout.flush()
print("\nDone.")
"""
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
for i in range(0, n_chars - seq_length, 1):
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

# build model
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
# load the network weights
filename = "weights-imporvement-08--2.0338.hdf5" # weights-iomporvement file with lowest loss value
model.load_weights(filename)
model.compile(loss="categorical_crossentropy", optimizer="adam")

# create mapping of unique characters to integers
int_to_char = dict((i, c) for i, c in enumerate(chars))


# pick a random seed
start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print("Seed:", "".join([int_to_char[value] for value in pattern]))
print("Pattern:", pattern)

sys.stdout.flush()
# generage characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    sys.stdout.flush()
    #print(x, prediction, index, result, seq_in, pattern)

model.summary()
score = model.evaluate(dataX, dataY, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
"""
