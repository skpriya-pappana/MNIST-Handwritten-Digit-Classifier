## MNIST Handwritten Digit Classifier

This project implements an Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset using scikit-learn's MLPClassifier.

It includes:
-Data visualization
-Preprocessing
-Model training and evaluation
-Predicting new digit images via user input
All built in a clean Object-Oriented Python class with proper exception handling.

Project Structure:
MNISTClassifier.py      # Main Python script
HD-Ouput Sample1.png   #Data visualization output
HD-Ouput Sample2.png   #Remaining output
The Dataset can be found at: https://infyspringboard.onwingspan.com/common-content-store/Shared/Shared/Public/lex_auth_0131395375426764803690_shared/web-hosted/assets/datasets1603947951084.zip

## Requirements
Make sure you have the following installed:
pip install numpy pandas matplotlib scikit-learn
How It Works:
1. Initialization:
The class loads the train.csv and test.csv files using pandas.
2. Display First Image:
Displays the first image in both datasets using matplotlib.
3. Preprocessing:
Splits features and labels from training data.
Fills any missing values (forward fill).
4. Model Training:
Trains a neural network with one hidden layer (50 nodes) and ReLU activation.
5. Evaluation:
It prints a classification report.
6. Predict New Inputs:
User can:
Enter one or more lines of comma-separated pixel values (784 or 785 integers).
785 → label + 784 pixels (label will be removed automatically)
Predicts each input line and prints the output digit.

## Sample Input Format


Each digit image must be a list of 784 pixel values (0–255) in a single line:
0,0,0,0,0,0,...,0   ← 784 numbers
Or if label is included:
5,0,0,0,...,0   ← 785 numbers (first number is label, will be ignored)
Type multiple such lines and then enter END when done.

## Example Usage (Interactive)

Do you want to predict using new input (yes/no)? yes
Enter each image input as a separate line of comma-separated pixel values (784 or 785 values).
When done, type END and press Enter.
0,0,0,0,0,...,0
0,0,0,0,0,...,0
END
Prediction 1: 2  
Prediction 2: 7  

## How to Run

Make sure train.csv and test.csv are in the correct path.
Run the script:
python MNISTClassifier.py
