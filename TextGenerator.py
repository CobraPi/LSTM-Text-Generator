import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras. models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding, CuDNNLSTM, CuDNNGRU, GRU
import numpy as np
import sys
import io
import os
import codecs
import random
import multiprocessing
from gensim.models import Word2Vec
import pandas as pd
import pickle

class TextGenerator:

    def __init__(self, checkpoint_name="", sequence_length=20, min_word_frequency=20, model_layers=1, step=1, batch_size=32, epochs=100, embedding=True):

        self.checkpoint_name = checkpoint_name
        self.inputfile = None
        self.outputfile = None

        self.corpus = ""
        self.vocabulary = set()
        self.vocabulary_size = 0
        self.word_indices = dict()
        self.indices_word = dict()
        self.word_frequency = dict()
        self.seed = []
        self.tokenizer = None

        self.text_in_words = []
        self.sentences = []
        self.sentences_test = []
        self.next_words_test = []
        self.next_words = []

        self.sent = []
        self.sent_list = []

        self.sequence_length = sequence_length
        self.min_word_frequency = min_word_frequency
        self.step = step
        self.batch_size = batch_size
        self.epochs = epochs

        self.embedding = embedding
        self.embedding_matrix = None
        self.embedding_size = 0
        self.model = None
        self.dropout = 0.2
        self.mem_cells = 512
        self.n_words = 0
        self.diversity_list = [0.3, 0.5, 0.6, 0.7, 1, 1.5]
        self.model_layers = model_layers

        self.w2v_model = None
        self.lowercase = True
        self.ignore_words = False
        self.min_words = 30
        self.max_words = 100



    def set_outpufile(self, outputfilepath):
        try:
            self.outputfile = open(outputfilepath, "w")
        except Exception as e:
            print("Exception raised-", str(e))

    def set_word_gen_range(self, min, max):
        self.min_words = min
        self.max_words = max



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
            if self.embedding:
                x = np.zeros((self.batch_size, self.sequence_length), dtype=np.int32)
                y = np.zeros((self.batch_size), dtype=np.int32)
            else:
                x = np.zeros((self.batch_size, self.sequence_length, len(self.vocabulary)), dtype=np.bool)
                y = np.zeros((self.batch_size, len(self.vocabulary)), dtype=np.bool)
            for i in range(self.batch_size):
                for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                    if self.embedding:
                        x[i, t] = self.word_indices[w]
                    else:
                        x[i, t, self.word_indices[w]] = 1
                if self.embedding:
                    y[i] = self.word_indices[next_word_list[index % len(sentence_list)]]
                else:
                    y[i, self.word_indices[next_word_list[index % len(sentence_list)]]] = 1
                index += 1
            yield x, y

    def get_model(self):
        if self.embedding:
            self.build_embedding_ml_model()
        else:
            self.build_ml_model()

    def train_w2v_model(self):
        print("Building Word2Vec")
        self.w2v_model = Word2Vec(sentences=self.text_in_words,
                             min_count=3,
                             window=5,
                             size=100,
                             sample=1e5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=20,
                             workers=multiprocessing.cpu_count())
        self.w2v_model.train(self.sentences, total_examples=self.w2v_model.corpus_count, epochs=30, report_delay=1)
        self.w2v_model.init_sims(replace=True)
        w2v_weights = self.w2v_model.wv.vectors
        self.vocabulary_size, self.embedding_size = w2v_weights.shape
        with open("models/" + self.checkpoint_name + "_w2v_model", "wb") as w2v:
            pickle.dump(self.w2v_model, w2v)

    def load_w2v_model(self):
        with open("models/" + self.checkpoint_name + "_w2v_model", "rb") as w2v:
            self.w2v_model = pickle.load(w2v)

    def build_ml_model(self):
        print("Building gru model...")
        self.model = Sequential()
        for i in range(self.model_layers):
            if i == self.model_layers - 1:
                return_seq = False
            else:
                return_seq = True
            self.model.add(Bidirectional(CuDNNGRU(self.mem_cells, return_sequences=return_seq), input_shape=(self.sequence_length, len(self.vocabulary))))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(len(self.vocabulary)))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()

    def build_embedding_ml_model(self):
        print("Building lstm embedding model...")

        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.vocabulary), output_dim=1024))
        for i in range(self.model_layers):
            if i == self.model_layers - 1:
                return_seq = False
            else:
                return_seq = True
            self.model.add(Bidirectional(CuDNNLSTM(self.mem_cells, return_sequences=return_seq)))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(len(self.vocabulary)))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()

    def load_saved_model(self, filepath):
        self.model = load_model(filepath)
        self.model.summary()

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probabs = np.random.multinomial(1, preds, 1)
        return np.argmax(probabs)

    def generate_text(self, diversity):
        sentence = self.seed
        div_string = "----- Diversity:" + str(diversity) + "\n\n"
        seed_string = '----- Generating with seed: "' + "\n" + ' '.join(sentence) + '"\n'
        text_string = "\n" + " ".join(sentence)
        gen_text = "\n\n" + "-----Generated Text: "
        print(div_string, end="")
        print(seed_string, end="")
        print(gen_text, end="")
        print(text_string, end="")
        if self.outputfile is not None:
            self.outputfile.write(div_string)
            self.outputfile.write(seed_string)
            self.outputfile.write(text_string)
        self.n_words = np.random.randint(self.min_words, self.max_words)
        for i in range(self.n_words):
            if self.embedding:
                x_pred = np.zeros((1, self.sequence_length))
            else:
                x_pred = np.zeros((1, self.sequence_length, len(self.vocabulary)))
            for t, word in enumerate(sentence):
                if self.embedding:
                    x_pred[0, t] = self.word_indices[word]
                else:
                    #print(self.word_indices[word])
                    x_pred[0, t, self.word_indices[word]] = 1.
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_word = self.indices_word[next_index]
            sentence = sentence[1:]
            sentence.append(next_word)
            n_word = " " + next_word
            print(n_word, end="")
            if self.outputfile is not None:
                self.outputfile.write(n_word)
        print("\n")
        if self.outputfile is not None:
            self.outputfile.write("\n \n")

    def on_epoch_end(self, epoch, logs):
        """
        Generates text at the end of each epoch
        :param epoch:
        :param logs:
        :return:
        """
        self.outputfile.write("\n----- Generating text after Epoch: %d\n" % epoch)
        seed_index = np.random.randint(len(self.sentences + self.sentences_test))
        self.seed = (self.sentences + self.sentences_test)[seed_index]
        for diversity in self.diversity_list:
            self.generate_text(diversity)
        line = "=" * 80 + "\n"
        print(line, end="")
        self.outputfile.write(line)
        self.outputfile.flush()

    def seed_in_vocabulary(self, seed):
        tokens = seed.split(" ")
        verified = True
        print("\n", end="")
        for word in tokens:
            if word not in self.vocabulary:
                print("'" + word + "'", "is NOT in vocabulary")
                verified = False
            else:
                print("'" + word + "'", "is in vocabulary")
        print("\n", end="")
        return verified

    def generate_text_on_run(self, seed="", user_seed=False):
        seed_index = np.random.randint(len(self.sentences + self.sentences_test))
        if not user_seed:
            self.seed = (self.sentences + self.sentences_test)[seed_index]
        else:
            self.seed = seed.split(" ")
        #diversity = np.random.randint(10, 200, 1) * 0.01
        self.generate_text(self.diversity_list[0])
        line = "=" * 80 + "\n"
        print(line, end="")
        if self.outputfile is not None:
            self.outputfile.write(line)
            self.outputfile.flush()

    def read_corpus_file(self, corpusfilename):
        try:
            with io.open(corpusfilename, encoding="utf-8") as f:
                self.corpus = f.read()
            print("Corpus length in characters:", len(self.corpus))
            if self.lowercase:
                self.corpus = self.corpus.lower()
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
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(self.sentences)

        except Exception as e:
            print("Exception raised -", str(e))

    def train(self):
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.log_device_placement = True
        #sess = tf.Session(config=config)
        #set_session(sess)
        #physical_devices = tf.config.list_physical_devices('GPU')
        #tf.config.experimental.set_memory_growth(physical_devices[0], enable=False)

        (self.sentences, self.next_words), (self.sentences_test, self.next_words_test) = self.shuffle_and_split_training_set(self.sentences, self.next_words)
        if not os.path.isdir('./checkpoints/'):
            os.makedirs('./checkpoints/')
        if self.embedding:
            file_path = "./checkpoints/" + self.checkpoint_name + "_LSTM_MODEL_EMBEDDING_" + str(self.model_layers) + "_LAYERS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                        "loss{loss:.4f}-val_loss{val_loss:.4f}-acc{acc:.4f}" % \
                        (len(self.vocabulary), self.sequence_length, self.min_word_frequency)
        else:
            file_path = "./checkpoints/" + self.checkpoint_name + "_LSTM_MODEL_" + str(self.model_layers) + "_LAYERS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                        "loss{loss:.4f}-val_loss{val_loss:.4f}-acc{acc:.4f}" % \
                        (len(self.vocabulary), self.sequence_length, self.min_word_frequency)
        checkpoint = ModelCheckpoint(file_path, monitor="acc", save_best_only=True)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor="acc", patience=20)
        callbacks_list = [checkpoint, print_callback, early_stopping]

        self.model.fit(self.generator(self.sentences, self.next_words),
                             steps_per_epoch=int(len(self.sentences) / self.batch_size) + 1,
                             epochs=self.epochs,
                             callbacks=callbacks_list,
                             validation_data=self.generator(self.sentences_test, self.next_words_test),
                             validation_steps=int(len(self.sentences_test) / self.batch_size) + 1)

    def save_model(self, filename):
        self.model.save(filename)
        if self.outputfile is not None:
            self.outputfile.close()
