import mlx.core as mx
import mlx.nn as nn
import random
random.seed(42)


# Hyperparameters
block_size = 3


# Load dataset
def load_dataset(filename):
    words = []
    with open(filename, 'r') as file:
        while line := file.readline():
            words.append(line.rstrip())
    return words


# Embed each character
def embed_chars(words):
    # Get unique chars in dataset
    chars = set()
    for word in words:
        for c in word:
            chars.add(c)

    # Embed each character
    stoi = {ch:i+1 for i, ch in enumerate(sorted(chars))}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    return (stoi, itos)


# Split into training and test
def split_dataset(words):
    X = []
    y = []
    for word in words:
        ctx = [0] * block_size
        for c in word + '.':
            idx = stoi[c]
            X.append(ctx)
            y.append(idx)
            ctx = ctx[1:] + [idx]
    return (mx.array(X), mx.array(y))


names = load_dataset('names.txt')
stoi, itos = embed_chars(names)

# Training, Dev, and Test sets
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
Xtr, Ytr = split_dataset(names[:n1])
Xdev, Ydev = split_dataset(names[n1:n2])
Xtest, Ytest = split_dataset(names[n2:])
print(f'Xtr.shape, Ytr.shape: [{Xtr.shape}, {Ytr.shape}]')
print(f'Xdev.shape, Ydev.shape: [{Xdev.shape}, {Ydev.shape}]')
print(f'Xtest.shape, Ytest.shape: [{Xtest.shape}, {Ytest.shape}]')