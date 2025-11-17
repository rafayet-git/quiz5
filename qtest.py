import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np
import argparse
import glob

path_text = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_text, 'rb').read().decode(encoding='utf-8')
chars = sorted(set(text))
vocab_size = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = np.array(chars)

text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)

seq_length = 100
examples_per_epoch = len(text_as_int) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_seq = chunk[:-1]      
    target_char = chunk[-1]     
    return input_seq, target_char

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

embedding_dim = 256
lstm_units = 512

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(lstm_units, return_sequences=True),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch'
)

def train_model(epochs=3):
    model.summary()
    return model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


def generate_text(seed_text, num_generate=200, temperature=1.0):
    """
    Generate text by iteratively predicting the next char given the previous `sequence_length` chars.
    - seed_text: initial string (can be shorter than sequence_length)
    - num_generate: how many characters to generate
    - temperature: controls randomness (higher -> more random)
    """
    # Prepare initial input ids (left-pad with index 0 if shorter than sequence_length)
    pad_idx = 0
    input_ids = [char2idx.get(c, pad_idx) for c in seed_text]
    if len(input_ids) < seq_length:
        input_ids = [pad_idx] * (seq_length - len(input_ids)) + input_ids
    else:
        input_ids = input_ids[-seq_length:]

    generated = []
    for _ in range(num_generate):
        x = np.array([input_ids], dtype=np.int32)  # shape (1, seq_length)
        preds = model.predict(x, verbose=0)        # shape (1, vocab_size)
        preds = preds[0].astype(np.float64)

        # Temperature scaling and sampling
        preds = np.log(preds + 1e-8) / max(1e-8, temperature)
        exp_preds = np.exp(preds)
        probs = exp_preds / np.sum(exp_preds)

        next_id = np.random.choice(range(vocab_size), p=probs)
        next_char = idx2char[next_id]
        generated.append(next_char)

        # Slide the window
        input_ids = input_ids[1:] + [next_id]
    # Return the seed plus generated continuation
    return seed_text + ''.join(generated)


def latest_checkpoint():
    weight_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")))
    return weight_files[-1] if weight_files else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate text with the LSTM model")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--generate', action='store_true', help='Generate text using the model')
    parser.add_argument('--seed', type=str, default='ROMEO:', help='Seed text for generation')
    parser.add_argument('--num_generate', type=int, default=400, help='Number of chars to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file to load')

    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs)

    if args.generate:
        weights_to_load = args.weights or latest_checkpoint()
        if weights_to_load:
            if not getattr(model, 'built', False):
                model.build((1, seq_length))
            model.load_weights(weights_to_load)
        else:
            print("No checkpoint weights found. If you want to generate with trained weights, run with --train first or provide --weights <path>.")
        sample = generate_text(args.seed, num_generate=args.num_generate, temperature=args.temperature)
        print("\n--- Generated sample ---\n")
        print(sample)
        print("\n--- End sample ---\n")

    if not args.train and not args.generate:
        parser.print_help()
