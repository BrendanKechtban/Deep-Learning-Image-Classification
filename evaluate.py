import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

model = tf.keras.models.load_model(os.path.join('saved_models', 'best_model.h5'))

(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255.0

y_test = tf.keras.utils.to_categorical(y_test, 10)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join('results', 'confusion_matrix.png'))
plt.show()

report = classification_report(y_true, y_pred_classes, output_dict=True)
print(report)

per_class_accuracy = {str(i): report[str(i)]['precision'] for i in range(10)}
np.save(os.path.join('results', 'per_class_accuracy.npy'), per_class_accuracy)
