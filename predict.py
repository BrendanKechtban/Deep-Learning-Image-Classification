import tensorflow as tf
import numpy as np
import pandas as pd
import os


model = tf.keras.models.load_model(os.path.join('saved_models', 'best_model.h5'))

(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255.0

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)


results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
predictions_df = pd.DataFrame({'TrueLabel': y_test, 'PredictedLabel': predicted_classes})
predictions_df.to_csv(os.path.join(results_dir, 'sample_predictions.csv'), index=False)
