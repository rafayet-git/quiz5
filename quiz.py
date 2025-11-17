import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np

# 1. Prepare text data
path_text = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_text, 'rb').read().decode(encoding='utf-8')

# Tokenize and create sequences (placeholder)
unique_characters = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(unique_characters)}
idx_to_char = {i: c for i, c in enumerate(unique_characters)}
sequence_length = 100

vocab_size = len(unique_characters) 
embedding_dim = 256
lstm_units = 512

# 2. Build LSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(lstm_units, return_sequences=True),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax'),
])

# 3. Train model (compile)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# 4. Generate text
def generate_text(seed_text, length=100, temperature=1.0, tokenizer=None):

    # Convert seed text to indices
    seed_indices = [char_to_idx.get(c, 0) for c in seed_text]

    # Ensure input length equals sequence_length by left-padding with 0s and truncating the left if too long
    if len(seed_indices) < sequence_length:
        seed_indices = [0] * (sequence_length - len(seed_indices)) + seed_indices
    else:
        seed_indices = seed_indices[-sequence_length:]

    def sample_from_probs(probs, temp):
        # Temperature scaling
        logits = np.log(np.asarray(probs, dtype=np.float64) + 1e-12) / temp
        exp = np.exp(logits - np.max(logits))
        scaled = exp / np.sum(exp)
        return int(np.random.choice(len(scaled), p=scaled))

    generated_chars = []
    input_seq = np.array(seed_indices, dtype=np.int32)

    for _ in range(int(length)):
        x = np.expand_dims(input_seq, axis=0)
        preds = model.predict(x, verbose=0)[0] 
        next_idx = sample_from_probs(preds, temperature)

        # Append next char and roll the input window
        generated_chars.append(idx_to_char.get(next_idx, ''))
        input_seq = np.concatenate([input_seq[1:], np.array([next_idx], dtype=np.int32)])

    return seed_text + ''.join(generated_chars)

print(generate_text("Shakespeare: ", length=200, temperature=0.8))