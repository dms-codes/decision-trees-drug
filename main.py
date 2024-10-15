import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Constant for the dataset file path
DRUG_FILE = "drug.csv"

def load_and_explore_data(file_path):
    """
    Load the dataset and display basic information, such as the first few rows and summary statistics.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    print("\nDataset Information:")
    df.info()
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by splitting features and labels, and encoding categorical variables.
    
    :param df: DataFrame containing the dataset
    :return: Encoded features X and labels y
    """
    # Split features (X) and labels (y)
    X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    y = df["Drug"]

    # Encode categorical features
    le_sex = preprocessing.LabelEncoder()
    X[:, 1] = le_sex.fit_transform(X[:, 1])  # Encode 'Sex' column

    le_BP = preprocessing.LabelEncoder()
    X[:, 2] = le_BP.fit_transform(X[:, 2])  # Encode 'BP' column

    le_Chol = preprocessing.LabelEncoder()
    X[:, 3] = le_Chol.fit_transform(X[:, 3])  # Encode 'Cholesterol' column

    return X, y

def split_train_test(X, y, test_size=0.3, random_state=3):
    """
    Split the dataset into training and testing sets.
    
    :param X: Features dataset
    :param y: Labels
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random seed for reproducibility
    :return: Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f'Train set: {X_train.shape}, {y_train.shape}')
    print(f'Test set: {X_test.shape}, {y_test.shape}')
    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train):
    """
    Build and train a Decision Tree classifier.
    
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained Decision Tree classifier
    """
    drug_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    drug_tree.fit(X_train, y_train)
    return drug_tree

def make_predictions(model, X_test):
    """
    Use the trained model to make predictions on the test set.
    
    :param model: Trained model
    :param X_test: Test features
    :return: Predictions made by the model
    """
    return model.predict(X_test)

def evaluate_predictions(y_test, y_pred):
    """
    Evaluate model performance by comparing predictions with actual values.
    
    :param y_test: Actual test labels
    :param y_pred: Predicted labels
    :return: Accuracy score
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Decision Tree's Accuracy: {accuracy:.4f}")
    return accuracy

def display_prediction_comparison(y_test, y_pred):
    """
    Display a comparison between predicted and actual values in a DataFrame.
    
    :param y_test: Actual test labels
    :param y_pred: Predicted labels
    :return: DataFrame showing predicted vs actual values
    """
    comparison_df = pd.DataFrame({'Predicted Values': y_pred, 'Actual Values': y_test})
    print("\nPrediction vs Actual Comparison:")
    print(comparison_df)
    return comparison_df

def main():
    # Load and preprocess the data
    df = load_and_explore_data(DRUG_FILE)
    X, y = preprocess_data(df)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Train the model
    drug_tree = build_and_train_model(X_train, y_train)

    # Make predictions
    y_pred = make_predictions(drug_tree, X_test)

    # Display prediction vs actual comparison
    display_prediction_comparison(y_test, y_pred)

    # Evaluate the model
    evaluate_predictions(y_test, y_pred)

if __name__ == '__main__':
    main()
