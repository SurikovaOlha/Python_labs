import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(input_file):
    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [int(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])

    X = np.array(X)
    y = np.array(y)

    return X, y

def plot_classifier(classifier, X, y, title='Classifier boundaries', annotate=False):
    # define ranges to plot the figure
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    step_size = 0.01

    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    mesh_output = mesh_output.reshape(x_values.shape)

    plt.figure()

    plt.title(title)

    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    plt.xticks(())
    plt.yticks(())

    if annotate:
        for x, y in zip(X[:, 0], X[:, 1]):
            plt.annotate(
                '(' + str(round(x, 1)) + ',' + str(round(y, 1)) + ')',
                xy = (x, y), xytext = (-15, 15),
                textcoords = 'offset points',
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.6', fc = 'white', alpha = 0.8),
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

def print_accuracy_report(classifier, X, y, num_validations=5):
    accuracy = train_test_split.cross_val_score(classifier,
            X, y, scoring='accuracy', cv=num_validations)
    print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

    f1 = train_test_split.cross_val_score(classifier,
            X, y, scoring='f1_weighted', cv=num_validations)
    print("F1: " + str(round(100*f1.mean(), 2)) + "%")

    precision = train_test_split.cross_val_score(classifier,
            X, y, scoring='precision_weighted', cv=num_validations)
    print("Precision: " + str(round(100*precision.mean(), 2)) + "%")

    recall = train_test_split.cross_val_score(classifier,
            X, y, scoring='recall_weighted', cv=num_validations)
    print("Recall: " + str(round(100*recall.mean(), 2)) + "%")