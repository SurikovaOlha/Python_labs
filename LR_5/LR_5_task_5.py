import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# y = 3x2+5

min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

plt.figure()
plt.scatter(data, labels)
plt.xlabel('Розмірність 1')
plt.ylabel('Розмірність 2')
plt.title('Вхідні дані')
plt.show()

# два приховані шари. Перший шар - 10 нейронів, другий шар - 6 нейронів, вихідний шар - 1 нейрон
nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])
nn.trainf = nl.train.train_gd

# тренування нейронної мережі
error_progress = nn.train(data, labels, epochs=2000, show=100, lr=0.01)

# виконання нейронної мережі
output = nn.sim(data)
y_pred = output.reshape(num_points)

# побудова графіка помилки навчання
plt.figure()
plt.plot(error_progress)
plt.xlabel('Кількість епох')
plt.ylabel('Помилка навчання')
plt.title('Зміна помилки навчання')
plt.show()

# побудова графіка результатів
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, ',', x, y_pred, 'p')
plt.title('Фактичні та прогнозовані значення')
plt.show()