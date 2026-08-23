# Insurance Charges Prediction using Linear Regression

## Project Overview

This project focuses on building a **Linear Regression Machine Learning model** to predict medical insurance charges based on customer-related information.

The project covers the complete basic Machine Learning workflow, starting from loading and understanding the dataset, performing data cleaning and preprocessing, exploratory data analysis, correlation analysis, preparing the data for Machine Learning, training a Linear Regression model, evaluating its performance, visualizing actual and predicted values, understanding model coefficients and intercept, and finally making predictions for new customers.

The main purpose of this project is to understand how different customer features can influence insurance charges and how Linear Regression can be used to make predictions based on these relationships.

---

## Objective

The main objectives of this project are:

- Load and understand the insurance dataset.
- Explore the structure and characteristics of the data.
- Check for missing values and possible data issues.
- Perform data cleaning and preprocessing.
- Understand the distribution of the data.
- Perform Exploratory Data Analysis (EDA).
- Analyze relationships between variables using correlation.
- Create a correlation heatmap.
- Identify important features related to insurance charges.
- Convert categorical data into numerical form where required.
- Define independent variables (X) and dependent variable (y).
- Split the dataset into training and testing data.
- Build a Linear Regression model.
- Train the model using training data.
- Make predictions on test data.
- Evaluate the model using MSE, RMSE, and R².
- Visualize Actual vs Predicted values.
- Understand model coefficients.
- Understand the model intercept.
- Make predictions for new customer data.

---

## Dataset

The project uses an insurance dataset containing information about customers and their medical insurance charges.

### Dataset Features

| Column | Description |
|---|---|
| `age` | Age of the customer |
| `sex` | Gender of the customer |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents |
| `smoker` | Smoking status of the customer |
| `region` | Residential region |
| `charges` | Medical insurance charges |

### Target Variable

The target variable for this project is:

`charges`

The model uses selected customer features to predict the value of `charges`.

---

## Features Used for the Model

The Linear Regression model uses the following features:

- `age`
- `bmi`
- `children`
- `smoker_encoded`

The target variable is:

- `charges`

Therefore:

`X = age, bmi, children, smoker_encoded`

`y = charges`

---

## Understanding the Features

### Age

`age` represents the age of the customer.

Age can have an effect on medical insurance charges because healthcare-related expenses can vary with age.

### BMI

`bmi` represents the Body Mass Index of the customer.

BMI is a numerical feature that can have a relationship with medical insurance charges.

### Children

`children` represents the number of children or dependents covered by the insurance plan.

The number of children can contribute to differences in insurance charges.

### Smoker

`smoker` represents whether the customer is a smoker.

Since Machine Learning models require numerical input, the categorical smoking feature was converted into numerical form.

For example:

`yes → 1`

`no → 0`

This resulted in the feature:

`smoker_encoded`

### Charges

`charges` is the target variable.

It represents the medical insurance charge for a customer.

The Linear Regression model attempts to predict this value using the selected input features.

---

## Machine Learning Workflow

The complete workflow followed in this project is:

Load Dataset  
↓  
Data Cleaning / Preprocessing  
↓  
Exploratory Data Analysis  
↓  
Correlation Analysis  
↓  
Feature Selection  
↓  
Define X and y  
↓  
Train-Test Split  
↓  
Create Linear Regression Model  
↓  
Train Model  
↓  
Make Predictions  
↓  
Calculate MSE  
↓  
Calculate RMSE  
↓  
Calculate R²  
↓  
Actual vs Predicted Visualization  
↓  
Analyze Coefficients  
↓  
Analyze Intercept  
↓  
New Data Prediction

---

## 1. Load Dataset

The insurance dataset is loaded using Pandas.

Pandas is used for loading, manipulating, cleaning, and analyzing tabular data.

Example:

`df = pd.read_csv("data/insurance.csv")`

After loading the dataset, the data is inspected to understand its structure.

---

## 2. Data Cleaning and Preprocessing

Data cleaning and preprocessing are important steps before applying a Machine Learning algorithm.

The dataset is checked for:

- Missing values
- Data types
- Duplicate records
- Invalid or inconsistent values
- Categorical variables
- Numerical variables

Basic Pandas operations can be used to inspect the dataset:

`df.head()`

Displays the first few rows.

`df.info()`

Provides information about columns, data types, and non-null values.

`df.describe()`

Provides statistical information about numerical columns.

Missing values can be checked using:

`df.isnull().sum()`

---

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset before building the Machine Learning model.

EDA helps answer questions such as:

- What does the dataset look like?
- What are the distributions of numerical features?
- Are there unusual values?
- Which features appear to have a relationship with charges?
- Does smoking status appear to affect charges?
- How are the numerical features related to each other?

EDA helps us understand the data before applying the model.

---

## 4. Correlation Analysis

Correlation analysis is used to understand the relationship between numerical variables.

Correlation values generally range from:

`-1 to +1`

### Positive Correlation

A positive correlation means that when one variable increases, the other variable tends to increase as well.

### Negative Correlation

A negative correlation means that when one variable increases, the other variable tends to decrease.

### Correlation Near Zero

A correlation close to zero indicates a weak linear relationship between the variables.

---

## 5. Correlation Heatmap

A correlation heatmap is used to visualize the relationships between numerical variables.

The heatmap makes it easier to identify which features have stronger or weaker relationships with `charges`.

One important observation from the analysis was that the smoking-related feature showed a strong relationship with insurance charges.

This suggests that smoking status is an important feature for predicting insurance charges.

However, correlation describes a relationship between variables and does not by itself prove causation.

---

## 6. Defining X and y

For Machine Learning, the dataset is divided into input features and the target variable.

### Independent Variables (X)

The input features used by the model are:

- `age`
- `bmi`
- `children`
- `smoker_encoded`

### Dependent Variable (y)

The target variable is:

`charges`

In simple terms:

Customer Information  
↓  
Linear Regression Model  
↓  
Predicted Insurance Charges

Example:

`X = df[["age", "bmi", "children", "smoker_encoded"]]`

`y = df["charges"]`

---

## 7. Train-Test Split

The dataset is divided into training and testing data.

The purpose of this split is to evaluate how well the model performs on data that it did not use during training.

Example:

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)`

### Training Data

Training data is used by the model to learn patterns and relationships between the input features and insurance charges.

### Testing Data

Testing data is kept separate and is used to evaluate the trained model.

The split used in this project is:

- 70% → Training Data
- 30% → Testing Data

---

## 8. Random State

The `random_state` parameter is used to make the train-test split reproducible.

`random_state=42`

Without specifying a random state, the data may be split differently every time the code runs.

Using a fixed value allows the same split to be reproduced.

---

## 9. Linear Regression

Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value.

In this project, the target variable is:

`insurance charges`

The model learns the relationship between the input features and the target variable.

The basic Linear Regression equation is:

`y = b0 + b1X1 + b2X2 + b3X3 + ... + bnXn`

Where:

- `y` = predicted insurance charges
- `b0` = intercept
- `b1, b2, ...` = coefficients
- `X1, X2, ...` = input features

For this project:

`Predicted Charges = Intercept + Age Coefficient × Age + BMI Coefficient × BMI + Children Coefficient × Children + Smoker Coefficient × Smoker`

---

## 10. Creating the Linear Regression Model

The Linear Regression model is created using Scikit-learn.

`from sklearn.linear_model import LinearRegression`

`model = LinearRegression()`

At this point, the model has been created but has not yet learned the patterns from the training data.

---

## 11. Training the Model

The model is trained using:

`model.fit(X_train, y_train)`

The `fit()` method allows the model to learn the relationship between:

`X_train → Input Features`

`y_train → Actual Charges`

During training, the model learns:

- Coefficients
- Intercept

These values are then used to make predictions.

---

## 12. Making Predictions

After training, predictions are made using:

`y_pred = model.predict(X_test)`

The model uses the input features from `X_test` and predicts insurance charges.

The predicted values can then be compared with the actual values from `y_test`.

---

## 13. Model Evaluation

The model is evaluated using three important metrics:

1. Mean Squared Error (MSE)
2. Root Mean Squared Error (RMSE)
3. R² Score

These metrics provide different information about the model's performance.

---

## 14. Mean Squared Error (MSE)

MSE stands for Mean Squared Error.

It calculates the average of the squared differences between actual and predicted values.

Conceptually:

Actual Charge  
↓  
Compare with Predicted Charge  
↓  
Calculate Error  
↓  
Square the Error  
↓  
Average all Errors  
↓  
MSE

A lower MSE generally indicates that the predictions are closer to the actual values.

The model achieved:

`MSE = 33,948,860`

---

## 15. Root Mean Squared Error (RMSE)

RMSE stands for Root Mean Squared Error.

It is calculated by taking the square root of MSE.

`RMSE = √MSE`

RMSE is easier to interpret because it is expressed in the same unit as the target variable.

The model achieved:

`RMSE = 5,826.56`

This means the prediction errors are typically on the order of several thousand charge units.

---

## 16. R² Score

R² is also called the coefficient of determination.

It measures how much of the variation in the target variable is explained by the model.

The model achieved:

`R² = 0.768`

or approximately:

`76.8%`

This means the model explains approximately 76.8% of the variation in insurance charges on the test data.

R² does not mean that every individual prediction is exactly 76.8% accurate.

It measures how well the model explains variation in the target variable.

---

## 17. MSE vs RMSE vs R²

| Metric | Meaning |
|---|---|
| MSE | Average squared prediction error |
| RMSE | Approximate size of prediction error in target units |
| R² | Amount of variation in the target explained by the model |

In simple terms:

`MSE → How large are the squared errors?`

`RMSE → How large are the errors approximately?`

`R² → How much variation does the model explain?`

---

## 18. Actual vs Predicted Graph

An Actual vs Predicted graph is used to compare:

Actual Insurance Charges  
vs  
Predicted Insurance Charges

The purpose of this graph is to visually understand how closely the predictions follow the actual values.

If predicted values are close to actual values, the model is performing better.

The graph can also help identify observations where the model's prediction is significantly different from the actual charge.

---

## 19. Model Coefficients

The model coefficients can be obtained using:

`print(model.coef_)`

A coefficient represents the change in predicted charges associated with a one-unit increase in a particular feature, while keeping the other features constant.

For example, if the coefficient of `age` is positive, increasing age by one unit would increase the predicted insurance charges by approximately the coefficient value, assuming the other features remain constant.

The smoking-related coefficient helps understand the effect of the encoded smoking feature on predicted charges.

Coefficients therefore help us understand how individual features influence the model's predictions.

---

## 20. Model Intercept

The intercept can be obtained using:

`print(model.intercept_)`

The intercept represents the model's predicted starting value when all input features are zero.

The Linear Regression equation is:

`Predicted Charges = Intercept + Age coefficient × Age + BMI coefficient × BMI + Children coefficient × Children + Smoker coefficient × Smoker`

The intercept is learned during model training.

The intercept does not necessarily represent a realistic customer's actual insurance charge because a customer with all input features equal to zero may not be a realistic case.

---

## 21. New Customer Prediction

Once the model has been trained, it can be used to predict charges for new customers.

Example:

`new_data = pd.DataFrame({
    "age": [25, 40, 50, 35, 60],
    "bmi": [22.5, 30.2, 28.5, 26.1, 32.0],
    "children": [1, 2, 3, 0, 2],
    "smoker_encoded": [0, 1, 0, 1, 0]
})`

Predictions can be generated using:

`new_predictions = model.predict(new_data)`

`print(new_predictions)`

The model will return estimated insurance charges for these customers.

These predictions are estimates based on patterns learned from the training data.

The actual charges may be higher or lower than the predicted values.

---

## 22. Why We Cannot Calculate R² for Unknown New Data

For the test dataset, we know both:

`Actual Charges + Predicted Charges`

Therefore, we can calculate:

- MSE
- RMSE
- R²

For completely new customers, we may only have:

Customer Information  
↓  
Model  
↓  
Predicted Charges

If the actual insurance charges are not known, we cannot calculate MSE, RMSE, or R² for those new observations.

For example:

New Customer  
↓  
Model  
↓  
Predicted Charge = ₹25,000

If we do not know the customer's real charge, we cannot determine the exact prediction error.

---

## 23. Model Results

The final Linear Regression model achieved the following results on the test dataset:

| Metric | Value |
|---|---:|
| MSE | 33,948,860 |
| RMSE | 5,826.56 |
| R² Score | 0.768 |

### Interpretation

The R² score of `0.768` indicates that approximately 76.8% of the variation in insurance charges is explained by the model on the test data.

The RMSE of `5,826.56` indicates that the prediction errors are on the order of several thousand charge units.

MSE represents the average squared prediction error.

---

## 24. Technologies Used

The project was developed using:

- **Python** — Programming language
- **Pandas** — Data loading, manipulation, cleaning, and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine Learning and model evaluation
- **Jupyter Notebook** — Interactive development and experimentation

---

## 25. Project Structure

Machine-Learning-Labs/  
└── Week3/  
    └── ML01/  
        ├── data/  
        │   └── insurance.csv  
        ├── notebook/  
        │   └── insurance_linear_regression.ipynb  
        ├── README.md  
        ├── requirements.txt  
        └── .gitignore

---

## 26. Requirements

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- jupyter

These dependencies are listed in:

`requirements.txt`

To install all required dependencies, run:

`pip install -r requirements.txt`

Jupyter is required to open and run the `.ipynb` notebook.

---

## 27. How to Run the Project

### Step 1: Clone the Repository

`git clone <repository-url>`

### Step 2: Navigate to the Project

`cd Machine-Learning-Labs/Week3/ML01`

### Step 3: Install Dependencies

`pip install -r requirements.txt`

### Step 4: Start Jupyter Lab

`jupyter lab`

### Step 5: Open the Notebook

Open:

`notebook/insurance_linear_regression.ipynb`

Run the notebook cells from top to bottom to reproduce the analysis and model.

---

## 28. Key Learnings

Through this project, the following Machine Learning concepts were practiced:

- Loading datasets using Pandas.
- Understanding dataset structure.
- Data cleaning and preprocessing.
- Handling categorical features.
- Encoding categorical data.
- Exploratory Data Analysis (EDA).
- Understanding numerical distributions.
- Correlation analysis.
- Creating and interpreting a correlation heatmap.
- Feature selection.
- Defining independent and dependent variables.
- Train-test splitting.
- Understanding training and testing data.
- Understanding `random_state`.
- Creating a Linear Regression model.
- Understanding `LinearRegression()`.
- Training a model using `fit()`.
- Making predictions using `predict()`.
- Understanding model coefficients.
- Understanding the model intercept.
- Evaluating predictions using MSE.
- Evaluating predictions using RMSE.
- Understanding and interpreting R².
- Comparing actual and predicted values.
- Visualizing model predictions.
- Making predictions on new data.
- Understanding why evaluation metrics require actual target values.

---

## 29. Conclusion

This project demonstrates the complete basic workflow of a supervised Machine Learning regression problem.

A Linear Regression model was trained using customer information to predict medical insurance charges.

The project started with dataset loading and preprocessing, followed by exploratory analysis and correlation analysis. The data was then divided into training and testing sets, and a Linear Regression model was trained using the training data.

The model was evaluated using MSE, RMSE, and R².

The model achieved an **R² Score of 0.768**, indicating that approximately **76.8% of the variation in insurance charges is explained by the model on the test dataset**.

The project also demonstrates that predictions made for completely new customers are estimates. If the actual charges for those customers are not known, evaluation metrics such as MSE, RMSE, and R² cannot be calculated for those predictions.

Overall, this project provides practical experience with:

**Data Cleaning → EDA → Correlation → Preprocessing → Train-Test Split → Linear Regression → Model Training → Prediction → Evaluation → Visualization → Coefficients → Intercept → New Data Prediction**
