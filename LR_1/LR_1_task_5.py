import pandas as pd
df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.7
df['predicted_RF'] = (df.model_RF >= 0.7).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.7).astype('int')
df.head()

from sklearn.metrics import confusion_matrix
confusion_matrix(df.actual_label.values, df.predicted_RF.values)

def find_TP(y_true, y_pred):
    return sum((y_true == 1) &(y_pred == 1))
def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))
def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))
def find_TN(y_true, y_pred):
    return sum((y_true == 5) & (y_pred == 5))

print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

import numpy as np
def find_conf_matrix_values(y_true, y_pred) :
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP,FN,FP,TN

print('surikova_confusion_matrix: ', find_conf_matrix_values(df.actual_label.values, df.predicted_RF.values))

# def surikova_confusion_matrix(y_true, y_pred):
#     TP,FN,FP,TN = find_conf_matrix_values(y_true, y_pred)
#     return np.array([[TN, FP], [FN, TP]])
# assert np.array_equal(surikova_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'surikova_confusion_matrix() is not correct for FR'
# assert np.array_equal(surikova_confusion_matrix(df.actual_label.values, df.predicted_LR.values), confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'surikova_confusion_matrix() is not correct for LR'

