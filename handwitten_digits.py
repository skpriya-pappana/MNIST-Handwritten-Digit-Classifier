import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

class MNISTClassifier:
    def __init__(self, train_path, test_path):
        try:
            self.mnist_train = pd.read_csv(train_path)
            self.mnist_test = pd.read_csv(test_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("File not found. Please check the file paths.")
        except Exception as e:
            print("Error loading data:", e)

    def display_first_images(self):
        try:
            train_digit = np.asarray(self.mnist_train.iloc[0, 1:]).reshape(28, 28)
            test_digit = np.asarray(self.mnist_test.iloc[0, :]).reshape(28, 28)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(train_digit, cmap=plt.cm.gray_r)
            plt.title("First image in training data")

            plt.subplot(1, 2, 2)
            plt.imshow(test_digit, cmap=plt.cm.gray_r)
            plt.title("First image in test data")
            plt.show()
        except Exception as e:
            print("Error displaying images:", e)

    def preprocess_data(self):
        try:
            self.x_train = self.mnist_train.iloc[:, 1:]
            self.y_train = self.mnist_train.iloc[:, 0]

            if 'label' in self.mnist_test.columns:
              self.x_test = self.mnist_test.iloc[:, 1:]
              self.y_test = self.mnist_test.iloc[:, 0]
            else:
              self.x_test = self.mnist_test.iloc[:, :]  # All columns are pixels
              self.y_test = None

            print("Filled missing values (if any) in x_train,y_train,x_test with Forward Fill.")

            self.x_train.fillna(method='ffill', inplace=True)#can use x_train.ffill() also
            self.y_train.fillna(method='ffill', inplace=True)

            self.x_test.fillna(method='ffill', inplace=True)

            print("Data preprocessing done.")

        except Exception as e:
            print("Error in preprocessing:", e)

    def train_model(self):
        try:
            self.model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', random_state=42)
            self.model.fit(self.x_train.values, self.y_train.values)
            print("Model training complete.")
        except Exception as e:
            print("Model training failed:", e)

    def predict_first(self):
        try:
            prediction = self.model.predict(self.mnist_test.iloc[0:1, :].values)
            print("Prediction for first test image:", prediction[0])
        except Exception as e:
            print("Prediction failed:", e)

    def evaluate_model(self):
        try:
            predicted_values = self.model.predict(self.x_train.values)
            report = classification_report(self.y_train.values, predicted_values)
            print("\nClassification Report:\n", report)
        except Exception as e:
            print("Evaluation failed:", e)

    def predict_new_input(self, new_data):

        try:
        # Convert input to NumPy array
            if isinstance(new_data, pd.DataFrame):
                data = new_data.values
            elif isinstance(new_data, list):
                data = np.array(new_data)
            else:
                data = new_data  # assume it's already a NumPy array

        # Handle single image input (1D)
            if data.ndim == 1:
                if data.shape[0] == 785:
                    print("Label column detected. Removing it.")
                    data = data[1:]
                if data.shape[0] != 784:
                    print("Invalid input: expected 784 features for one image.")
                    return
                data = data.reshape(1, -1)  # reshape to (1, 784)

        # Handle multiple images (2D)
            elif data.ndim == 2:
                if data.shape[1] == 785:
                    print("Label column detected. Removing it.")
                    data = data[:, 1:]
                if data.shape[1] != 784:
                    print("Invalid input: each row must have 784 pixel values.")
                    return

            else:
                print("Invalid input: expected 1D or 2D data.")
                return

        # Predict using the trained model
            predictions = self.model.predict(data)

        # Number and print each prediction using enumerate
            for i, pred in enumerate(predictions):
                print(f"Prediction {i+1}: {pred}")

            return predictions

        except Exception as e:
            print("Error during prediction:", e)


# 🔧 Main Execution
if __name__ == "__main__":
    classifier = MNISTClassifier("C:\\Users\\Nithin\\Documents\\Projects\\Handwritten digit recognition\\train.csv", "C:\\Users\\Nithin\\Documents\\Projects\\Handwritten digit recognition\\test.csv")
    classifier.display_first_images()
    classifier.preprocess_data()
    classifier.train_model()
    classifier.predict_first()
    classifier.evaluate_model()
    try:
        choice = input("Do you want to predict using new input (yes/no)? ").lower()
        while choice == 'yes':
            print("Enter each image input as a separate line of comma-separated pixel values (784 or 785 values).")
            print("When done, type END and press Enter.")

            lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line.strip())

        # Convert all lines into 2D list
            new_inputs = []
            for line in lines:
                values = [int(val.strip()) for val in line.split(',')]
                new_inputs.append(values)

        # Send to prediction
            classifier.predict_new_input(new_inputs)

            choice = input("Do you want to predict using new input (yes/no)? ").lower()
    except Exception as e:
      print("Error during prediction:", e)