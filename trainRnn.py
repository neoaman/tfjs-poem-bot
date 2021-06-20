# Import required packages
"""
If some packages are unavailable you need to download it manually
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflowjs as tfjs
import json


# Defining some asserts which need modifications based on choice
text_file_location = "public/static/poem.txt"
modeltarget_dir = "public/static/model"
epoch = 500


textfile = open(text_file_location).read()

corpus = textfile.lower().replace("\n","0<|>").split("<|>")



tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus,)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index

# create sequences
input_sequences = []
for line in corpus:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_gram_sequence = tokens[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_seq_len = max([len(i) for i in input_sequences])
input_seq_array = np.array(pad_sequences(input_sequences,
                                         maxlen=max_seq_len,
                                         padding='pre')
                        )
# segrating features(X) and label(y)
X = input_seq_array[:, :-1]
labels = input_seq_array[:, -1]
y = tf.keras.utils.to_categorical(labels, num_classes=vocab_size) # One hot encoding

# The model
model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, 120, input_length=max_seq_len-1),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(120)),
                tf.keras.layers.Dense(vocab_size, activation='softmax')
])
##define the learning rate - step size for optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(X, y, epochs=epoch, verbose=1)

# Get word token from word index ( Just reverse the key value pair)
word_token = {j:i for i,j in word_index.items()}
word_token["max_seq_len"] = max_seq_len

# Dump the objects/dictonary to corresponding json files
with open(modeltarget_dir+'/word_token.json','w') as wj:
    json.dump(word_token,wj)
with open(modeltarget_dir+'/word_index.json','w') as wj:
    json.dump(word_index,wj)

# Convert keras model to tfjs model
tfjs.converters.save_keras_model(model,modeltarget_dir )

if __name__=="__main__":
    seed_text = "Which fingure did he bite"
    next_words = 10
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        output_word = word_token[predicted[0]]
        if "0" in output_word[-1]: 
            output_word = output_word[:-1]
            seed_text += " " + output_word +"."
            break
        seed_text += " " + output_word
    print(seed_text)
