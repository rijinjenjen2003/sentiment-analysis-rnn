sentences = [
    "I love this product",
    "This is amazing",
    "I hate this",
    "This is bad"
]

labels = [1, 1, 0, 0]

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)
print(tokenizer.word_index) 
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences, padding='post')

print(padded)
padded = np.array(padded, dtype=np.int32)
labels = np.array(labels, dtype=np.float32)
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=50, output_dim=8),
    tf.keras.layers.SimpleRNN(8),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(padded, labels, epochs=10)

test = ["I love this"]
test_seq = tokenizer.texts_to_sequences(test)
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1], padding='post')

prediction = model.predict(test_pad)

print("Prediction:", prediction)