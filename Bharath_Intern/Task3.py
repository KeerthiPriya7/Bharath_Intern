import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


iris = datasets.load_iris()
X = iris.data
y = iris.target

class_labels = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy}")

mlp_model = MLPClassifier(max_iter=1000)
mlp_model.fit(X_train, y_train)
mlp_predictions = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f"MLP Accuracy: {mlp_accuracy}")

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

nn_accuracy = model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {nn_accuracy}")

new_object = np.array([[6.4,3.2,5.3,2.3]])  
new_object_standardized = scaler.transform(new_object)

svm_prediction = svm_model.predict(new_object_standardized)
svm_predicted_class = label_encoder.inverse_transform(svm_prediction)
print(f"SVM Predicted Class: {svm_predicted_class}")

mlp_prediction = mlp_model.predict(new_object_standardized)
mlp_predicted_class = label_encoder.inverse_transform([np.argmax(mlp_prediction)])
print(f"MLP Predicted Class: {mlp_predicted_class}")

nn_prediction = model.predict(new_object_standardized)
nn_predicted_class = label_encoder.inverse_transform(nn_prediction.argmax(axis=1))
print(f"Neural Network Predicted Class: {nn_predicted_class}")

class_labels = iris.target_names

svm_predicted_class_label = class_labels[svm_predicted_class[0]]
mlp_predicted_class_label = class_labels[mlp_predicted_class[0]]
nn_predicted_class_label = class_labels[nn_predicted_class[0]]

print(f"SVM Predicted Class Label: {svm_predicted_class_label}")
print(f"MLP Predicted Class Label: {mlp_predicted_class_label}")
print(f"Neural Network Predicted Class Label: {nn_predicted_class_label}")

