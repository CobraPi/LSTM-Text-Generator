from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras. models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import sys
import io
import os
import codecs


class TwitterBot:

    def __init__(self, sequence_length=10, min_word_frequency=20, step=1, batch_size=32, epochs=100, model_layers=1):
        self.inputfile = None
        self.outputfile = None

        self.corpus = ""
        self.vocabulary = set()
        self.word_indices = dict()
        self.indices_word = dict()
        self.word_frequency = dict()

        self.text_in_words = []
        self.sentences = []
        self.sentences_test = []
        self.next_words = []

        self.sequence_length = sequence_length
        self.min_word_frequency = min_word_frequency
        self.step = step
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = None
        self.dropout = 0.2
        self.model_layers = model_layers
        self.n_words = 50
        self.diversity_list = [0.3, 0.4, 0.5, 0.6, 0.7]

        self.ignore_words = False

    def set_outpufile(self, outputfilepath):
        try:
            self.outputfile = open(outputfilepath, "w")
        except Exception as e:
            print("Exception raised-", str(e))

    def shuffle_and_split_training_set(self, sentences_original, next_original, percentage_test=2):
        print("Shuffling sentences")
        tmp_sentences = []
        tmp_next_word = []
        for i in np.random.permutation(len(sentences_original)):
            tmp_sentences.append(sentences_original[i])
            tmp_next_word.append(next_original[i])
        cut_index = int(len(sentences_original) * (1. - (percentage_test/100)))
        x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
        y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]
        print("Sise of training set = %d" % len(x_train))
        print("Size of test set = %d" % len(y_test))
        return (x_train, y_train), (x_test, y_test)

    def generator(self, sentence_list, next_word_list):
        index = 0
        while True:
            x = np.zeros((self.batch_size, self.sequence_length, len(self.vocabulary)), dtype=np.bool)
            y = np.zeros((self.batch_size, len(self.vocabulary)), dtype=np.bool)
            for i in range(self.batch_size):
                for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                    x[i, t, self.word_indices[w]] = 1
                y[i, self.word_indices[next_word_list[index % len(sentence_list)]]] = 1
                index += 1
            yield x, y

    def set_model_layers(self, model_layers):
        self.model_layers = model_layers

    def build_model(self, model_layers):
        print("Building model...")
        self.model = Sequential()
        if model_layers == 1:
            self.model.add(Bidirectional(LSTM(128), input_shape=(self.sequence_length, len(self.vocabulary))))
        else:
            self.model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.sequence_length, len(self.vocabulary))))
        self.model.add(Dropout(self.dropout))
        for i in range(model_layers - 1):
            self.model.add(Bidirectional(LSTM(128)))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(len(self.vocabulary)))
        self.model.add(Activation("softmax"))

    def load_saved_model(self, filepath):
        self.model = load_model(filepath)

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probabs = np.random.multinomial(1, preds, 1)
        return np.argmax(probabs)

    def on_epoch_end(self, logs, epoch):
        self.outputfile.write("\n----- Generating text after Epoch: %d\n" % epoch)
        seed_index = np.random.randint(len(self.sentences + self.sentences_test))
        seed = (self.sentences + self.sentences_test)[seed_index]
        for diversity in self.diversity_list:
            sentence = seed
            div_string = "----- Diversity:" + str(diversity) + "\n"
            seed_string = '----- Generating with seed:\n"' + ' '.join(sentence) + '"\n'
            text_string = " ".join(sentence)
            print(div_string)
            print(seed_string)
            print(text_string)
            self.outputfile.write(div_string)
            self.outputfile.write(seed_string)
            self.outputfile.write(text_string)
            for i in range(self.n_words):
                x_pred = np.zeros((1, self.sequence_length, len(self.vocabulary)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, self.word_indices[self.vocabulary]] = 1.
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_word = self.indices_word[next_index]
                sentence = sentence[1:]
                sentence.append(next_word)
                n_word = " " + next_word
                print(n_word)
                self.outputfile.write(n_word)
            print("")
            self.outputfile.write("\n")
        line = "=" * 80 + "\n"
        print(line, end="")
        self.outputfile.write(line)
        self.outputfile.flush()

    def read_corpus(self, corpusfilename):
        try:
            with io.open(corpusfilename, encoding="utf-8") as f:
                self.corpus = f.read()
            print("Corpus length in characters:", len(self.corpus))
            self.text_in_words = [w for w in self.corpus.split(" ") if w.strip() != "" or w == "\n"]
            print("Corpus length in words:", len(self.text_in_words))
            for word in self.text_in_words:
                self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
            self.vocabulary = set(self.text_in_words)
            if self.ignore_words:
                ignored_words = set()
                for k, v in self.word_frequency.items():
                    if self.word_frequency[k] < self.min_word_frequency:
                        ignored_words.add(k)
                print("Unique words before ignoring:", len(self.vocabulary))
                print("Ignoring words with frequency <", self.min_word_frequency)
                self.vocabulary = sorted(set(self.vocabulary) - ignored_words)
                print("Unique words after ignoring", len(self.vocabulary))
                self.word_indices = dict((c, i) for i, c in enumerate(self.vocabulary))
                self.indices_word = dict((i, c) for i, c in enumerate(self.vocabulary))
                ignored = 0
                for i in range(0, len(self.text_in_words) - self.sequence_length, self.step):
                    if len(set(self.text_in_words[i: i+self.sequence_length + 1]).intersection(ignored_words)) == 0:
                        self.sentences.append(self.text_in_words[i: i + self.sequence_length])
                        self.next_words.append(self.text_in_words[i + self.sequence_length])
                print("Ignored sequences:", ignored)
                print("Remaining sequences:", len(self.sentences))
            else:
                self.vocabulary = sorted(set(self.vocabulary))
                self.word_indices = dict((c, i) for i, c in enumerate(self.vocabulary))
                self.indices_word = dict((i, c) for i, c in enumerate(self.vocabulary))
                for i in range(0, len(self.text_in_words) - self.sequence_length, self.step):
                    self.sentences.append(self.text_in_words[i: i + self.sequence_length])
                    self.next_words.append(self.text_in_words[i + self.sequence_length])
        except Exception as e:
            print("Exception raised -", str(e))

    def train(self):
        (self.sentences, self.next_words), (self.sentences_test, next_words_test) = self.shuffle_and_split_training_set(self.sentences, self.next_words)
        if not os.path.isdir('./checkpoints/'):
            os.makedirs('./checkpoints/')
        file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                    "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
                    (len(self.vocabulary), self.sequence_length, self.min_word_frequency)
        checkpoint = ModelCheckpoint(file_path, monitor="val_acc", save_best_only=False)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor="val_acc", patience=5)
        callbacks_list = [checkpoint, print_callback, early_stopping]

        self.model.fit_generator(self.generator(self.sentences, self.next_words, self.batch_size),
                                 steps_per_epoch=int(len(self.sentences) / self.batch_size) + 1,
                                 epochs=self.epochs,
                                 callbacks=callbacks_list,
                                 validation_data=self.generator(self.sentences_test, next_words_test, self.batch_size),
                                 validation_steps=int(len(self.sentences_test) / self.batch_size) + 1)


    def save_model(self, filename):
        self.model.save(filename)
        self.outputfile.close()

