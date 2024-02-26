import mlx.core as mx
import mlx.nn as nn
import numpy as np
import random

# Seed random generators for reproducibility
mx.random.seed(42)
random.seed(42)


# Hyperparameters
block_size = 3
C_size = 2
W1_size = 300
training_iters = 30000


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


# Loss function -- passing all trainable-parameters to calculate the gradient
# NOTE: This includes the forward pass
def loss_fn(params, ix):
    emb_vals = mx.flatten(C[Xtr[ix]], start_axis=1)
    h = mx.tanh(emb_vals @ params['W1'] + params['b1'])
    logits = h @ params['W2'] + params['b2']
    return nn.losses.cross_entropy(logits, Ytr[ix], reduction='mean')


# Function to perform forward pass, calculating loss and gradient
loss_and_grad_fn = mx.value_and_grad(loss_fn)


# Sample from the model
def sample(num_samples, params):
    for _ in range(num_samples):
        out = []
        context = [0] * block_size
        while True:
            emb_vals = mx.flatten(params['C'][mx.array(context)])
            h = mx.tanh(emb_vals @ params['W1'] + params['b1'])
            logits = h @ params['W2'] + params['b2']
            probs = mx.softmax(logits)
            np_probs = np.array(probs)

            # TODO: The following function casts the values to float64 before
            # sampling, resulting in values whose sum > 1. In this situation,
            # the function throws an error and the sampling fails. So, this 
            # must be fixed!
            ixList = np.random.multinomial(1, np_probs)
            ix = np.where(ixList == 1)[0].item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0: break
        
        # Print result
        print(''.join(itos[i] for i in out))


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

# Define the neural network hyperparameters
C = mx.random.normal([len(itos), C_size])
W1 = mx.random.normal([block_size * C_size, W1_size])
b1 = mx.random.normal([W1_size])
W2 = mx.random.normal([W1_size, len(itos)])
b2 = mx.random.normal([len(itos)])
trainable_params = {'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# Training loop
for i in range(training_iters):

    # Use a minibatch
    ix = mx.random.randint(0, Xtr.shape[0], (32,))

    # Forward pass, collecting loss and gradients
    loss, grads = loss_and_grad_fn(trainable_params, ix)

    # Update parameters
    for k in trainable_params.keys():
        trainable_params[k] += -0.01 * grads[k]

    # Print training update
    if (i + 1) % 1000 == 0: print(f'Loss at iteration {i + 1}:', loss.item())
    # Print final loss
    if i == training_iters - 1: print('Loss on last training iteration:', loss.item())


# Evaluate loss on dev set
emb_vals = mx.flatten(C[Xdev], start_axis=1)
h = mx.tanh(emb_vals @ trainable_params['W1'] + trainable_params['b1'])
logits = h @ trainable_params['W2'] + trainable_params['b2']
dev_loss = nn.losses.cross_entropy(logits, Ydev, reduction='mean').item()
print('Dev Loss:', dev_loss)

# Sample 20 names
sample(20, trainable_params)