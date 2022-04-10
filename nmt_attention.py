import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

use_builtins = True
punctuation = '.?!,'
acceptable_chars = punctuation + ' abcdefghijklmnopqrstuvwxyzабвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
acceptable_chars = tensorflow_text.normalize_utf8(acceptable_chars, 'NFKD')


def load_data(path):
  text = open(path, 'r', encoding='utf-8').read()

  lines = text.splitlines()
  pairs = [line.split('\t') for line in lines]

  inp = [inp.lower() for targ, inp, _ in pairs]
  targ = [targ.lower() for targ, inp, _ in pairs]

  return inp, targ


def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tensorflow_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^' + acceptable_chars + ']', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[' + punctuation + ']', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


# get raw data
path_to_data = 'ukr-eng/ukr.txt'
inp, targ = load_data(path_to_data)

# tf.data.Dataset
BUFFER_SIZE = len(inp) # whole dataset size
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

# text vectorization
max_vocab_size = 5000

input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)
input_text_processor.adapt(inp)

for example_input_batch, example_target_batch in dataset.take(1):
  example_tokens = input_text_processor(example_input_batch)
  input_vocab = np.array(input_text_processor.get_vocabulary())
  tokens = input_vocab[example_tokens[0].numpy()]
  print(' '.join(tokens))


output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

output_text_processor.adapt(targ)
print(output_text_processor.get_vocabulary()[:10])
output_text_processor.adapt(inp)
print(output_text_processor.get_vocabulary()[:10])
