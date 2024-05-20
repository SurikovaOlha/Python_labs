import numpy as np
import random

import rnn

from rnn import RNN
from data import train_data, test_data

from sklearn.utils.extmath import softmax

train_data = {
    'good': True,
    'bad': False,
}

test_data = {
    'this is happy': True,
    'i am good': True,
}

from data import train_data, test_data

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
print(word_to_idx['good'])
print(idx_to_word[0])

def createInputs(text):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)

    return inputs

class RNN:
    # Классическая рекуррентная нейронная сеть

    def __init__(self, input_size, output_size, hidden_size=64):
        self.Whh = random(hidden_size, hidden_size) / 1000
        self.Wxh = random(hidden_size, input_size) / 1000
        self.Why = random(output_size, hidden_size) / 1000

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def softmax(xs):
        # Застосування функції Softmax для вхідного масиву
        return np.exp(xs) / sum(np.exp(xs))

    # ініціалізація нейронної мережі
    rnn = RNN(vocab_size, 2)

    inputs = createInputs('i am very good')
    out, h = rnn.forward(inputs)
    probs = softmax(out)
    print(probs)

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Выполнение каждого шага нейронной сети RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Подсчет вывода
        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)

        # Підрахунок
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Ініціалізація до нуля
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Обчислення до останнього h
        d_h = self.Why.T @ d_y

        # Зворотнє розповсюдження по часу
        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T
            d_h = self.Whh @ temp

        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Оновлюємо ваги і зміщення
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh

        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

    # Цикл для кожного прикладу тренування
    for x, y in train_data.items():
        inputs = createInputs(x)
        target = int(y)

        # Пряме разповсюдження
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Створення dL/dy
        d_L_d_y = probs
        d_L_d_y[target] -= 1

        rnn.backprop(d_L_d_y)

    # Допоміжна функція
    def processData(data, backprop=True):
        items = list(data.items())
        random.shuffle(items)

        loss = 0
        num_correct = 0

        for x, y in items:
            inputs = createInputs(x)
            target = int(y)

            # Forward
            out, _ = rnn.forward(inputs)
            probs = softmax(out)

            # Calculate loss / accuracy
            loss -= np.log(probs[target])
            num_correct += int(
                np.argmax(probs) == target)

            if backprop:
                # Build dL/dy
                d_L_d_y = probs
                d_L_d_y[target] -= 1

                # Backward
                rnn.backprop(d_L_d_y)

        return loss / len(data), num_correct / len(data)

    for epoch in range(1000):
        train_loss, train_acc = processData(train_data)

        if epoch % 100 == 99:
            print('--- Epoch %d' % (epoch + 1))
            print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

            test_loss, test_acc = processData(test_data, backprop=False)
            print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))