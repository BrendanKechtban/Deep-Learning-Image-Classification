# Fashion MNIST CNN Classifier

This project is a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images from the Fashion MNIST dataset.

## Project Structure
- `train.py`: Train the CNN model and save the best model.
- `predict.py`: Use the trained model to make predictions on the test set and export results.
- `evaluate.py`: Evaluate the model, generate a confusion matrix, and save per-class accuracy.
- `requirements.txt`: List of required Python packages.


## Setup
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd fashion-mnist-cnn
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Train the model:**
   ```sh
   python train.py
   ```
2. **Make predictions:**
   ```sh
   python predict.py
   ```
3. **Evaluate the model:**
   ```sh
   python evaluate.py
   ```

## Confusion Matrix

<img width="1190" height="822" alt="confusion_matrix_photo" src="https://github.com/user-attachments/assets/054578c5-9db0-4687-9863-3da675ef67d8" />

