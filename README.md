# Drug Classification Using Decision Tree

This project implements a Decision Tree Classifier to predict the type of drug a patient should take based on their characteristics (age, sex, blood pressure, cholesterol levels, and sodium-potassium ratio). The model is trained using a dataset and evaluated based on its accuracy.

## Features

- **Data Loading and Exploration**: Loads the dataset and displays basic statistics and information.
- **Preprocessing**: Encodes categorical variables (Sex, BP, Cholesterol) into numerical values.
- **Model Training**: A Decision Tree classifier is trained on the dataset.
- **Prediction**: The trained model is used to predict drug type on the test data.
- **Evaluation**: The model’s accuracy is evaluated and predictions are compared against actual values.

## Project Structure

```bash
.
├── main.py          # Main script for running the project
├── drug.csv         # Dataset used for training and testing the model
└── README.md        # Project documentation
```

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python libraries:
  - pandas
  - scikit-learn
  - matplotlib

You can install the necessary dependencies using `pip`:

```bash
pip install pandas scikit-learn matplotlib
```

## Dataset

The dataset `drug.csv` contains the following columns:

- `Age`: Age of the patient
- `Sex`: Gender of the patient (M/F)
- `BP`: Blood pressure levels (LOW, NORMAL, HIGH)
- `Cholesterol`: Cholesterol levels (NORMAL, HIGH)
- `Na_to_K`: Sodium-Potassium ratio in the patient's blood
- `Drug`: The type of drug to be prescribed (target label)

Make sure the dataset file `drug.csv` is in the same directory as `main.py` or provide the correct path to it.

## How to Run the Project

1. Clone the repository or download the source code.
2. Ensure the `drug.csv` file is available in the same directory as `main.py`.
3. Run the main Python script:

```bash
python main.py
```

This will:
- Load and preprocess the data.
- Split the data into training and testing sets.
- Train a Decision Tree model on the training data.
- Evaluate the model's performance on the test data.
- Display a comparison of predicted and actual values.

### Output

- The script will print information about the dataset, such as the first 5 rows, statistical summary, and data shape.
- After training the model, it will print the accuracy of the model.
- It will also display a DataFrame comparing the predicted and actual drug values.

## Example Output

```bash
Dataset Information:
<class 'pandas.core.frame.DataFrame'>
...

First 5 Rows of the Dataset:
   Age Sex     BP Cholesterol  Na_to_K   Drug
0   23   F   HIGH        NORMAL    25.355  drugY
...

Statistical Summary:
...

Train set: (140, 5) (140,)
Test set: (60, 5) (60,)

Prediction vs Actual Comparison:
    Predicted Values Actual Values
0             drugX         drugY
1             drugY         drugY
...

Decision Tree's Accuracy: 0.85
```

## License

This project is licensed under the MIT License.

---

### Customization

You can extend this project by:
- Tuning the Decision Tree parameters (e.g., changing the criterion or maximum depth).
- Trying different machine learning models such as Random Forest or Logistic Regression.
- Adding more preprocessing steps if necessary.

