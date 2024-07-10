# Pre-Interview Assessment Classification

## Introduction
This project aims to classify candidates based on their pre-interview assessment data to predict if they will be accepted for the interview. The dataset includes various features about the candidates, and the target variable is the 'accepted for the interview' column. The project involves data preprocessing, visualization, model training using Logistic Regression, and evaluation.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- OrdinalEncoder

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Station-Project-2-A-Classification-Challenge.git
    ```
2. Navigate to the project directory:
    ```bash
    cd pre_interview_assessment_
    ```

## Running the Project

### Running Locally
1. Ensure all required libraries are installed.
2. Download the dataset via this link (https://www.kaggle.com/datasets/raneemrefaie/pre-interview-acceptance?resource=download)
3. Open the project file (`pre_interview_assessment_classification.py`) in your preferred IDE or text editor.
4. Run the script to execute the code, visualize data, and evaluate the model.

### Running on Google Colab
1. Open the provided Google Colab link in the main code (7th row in `pre_interview_assessment_.py` project directory).
2. Run all cells to see the visualizations and evaluations.

## Libraries and Functions Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For statistical data visualization.
- **scikit-learn**: For machine learning algorithms, model evaluation metrics, and data preprocessing (e.g., `OrdinalEncoder`).

## Project Steps and Outputs

### 1. Data Description
Described the dataset to understand its statistical properties.

Example output:
```python
# Statistical description
print(df.describe())
```

### 2. Data Correlation
Generated a correlation matrix to understand the relationships between different features.

Example:
```python
# Correlation matrix
print(df.corr())
```

### 3. Data Preprocessing
Encoded categorical features using `OrdinalEncoder`.

Example:
```python
from sklearn.preprocessing import OrdinalEncoder

# Encoding categorical columns
encoder = OrdinalEncoder()
df_encoded = df.copy()
df_encoded[categorical_columns] = encoder.fit_transform(df[categorical_columns])
```

### 4. Data Splitting and Model Training
Split the data into features (X) and target (y), then trained the Logistic Regression model.

Example:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Splitting the data
X = df_encoded.drop('accepted for the interview', axis=1)
y = df_encoded['accepted for the interview']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5. Model Evaluation
Computed accuracy, F1 score, recall, precision, and confusion matrix with its heatmap.

Example:
```python
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Results
The model's evaluation metrics are as follows:
- **Accuracy**: 0.9063545150501672
- **F1 Score**: 0.9063545150501672
- **Recall**: 0.9063545150501672
- **Precision**: 0.9063545150501672

These results demonstrate the classification capabilities of the Logistic Regression model for predicting if a candidate will be accepted for the interview based on pre-interview assessment data.

## Conclusion
This project demonstrates the process of classifying candidates using pre-interview assessment data and machine learning techniques. By following the steps provided, you can run and test the model, visualize the data, and evaluate the model's performance.
