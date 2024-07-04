

# Machine Learning Algorithms Overview 🚀 Mindmap

## **Linear Regression** 📈

### **Introduction**
- **Definition**: A method to model the relationship between a dependent variable and one or more independent variables.
- **Goal**: Predict the dependent variable based on the independent variables.

### **Mind Map**
![Linear Regression Mind Map](https://github.com/ParimalA24-DS/MACHINE-LEARNING-ALGORITHM-MINDMAP/blob/main/MLMINDMAPS/1.LINEARREGRESSION_MINDMAP.PNG)

### **Types**
- **Simple Linear Regression**
  - Predicting a dependent variable based on one independent variable.
  - **Example**: Predicting house prices based on square footage.
- **Multiple Linear Regression**
  - Predicting a dependent variable based on multiple independent variables.
  - **Example**: Predicting house prices based on square footage, bedrooms, and location.

### **Key Concepts**
- **Dependent Variable (Y)**: The variable to predict.
- **Independent Variables (X)**: Variables used for prediction.
- **Intercept (β₀)**: Expected value of Y when X is zero.
- **Slope (β₁)**: Change in Y for a one-unit change in X.

### **Formula**
- **Equation**: \( Y = β₀ + β₁ \cdot X + \epsilon \)
  - Represents the linear relationship between X and Y, with \( \epsilon \) as the error term.

### **Steps**
1. **Collect Data**: Gather data for dependent and independent variables.
2. **Fit the Model**: Find the best-fitting line using statistical techniques.
3. **Evaluate**: Assess model performance with metrics like R-squared.
4. **Predict**: Use the model to make predictions.

### **Applications**
- **Finance**: Predicting stock prices from historical data.
- **Marketing**: Estimating sales from advertising expenditure.
- **Healthcare**: Predicting patient outcomes from medical features.

### **Advantages**
- **Simple and Interpretable**: Easy to understand and explain.
- **Efficient**: Computationally inexpensive.
- **Provides Insights**: Understand relationships between variables.

### **Disadvantages**
- **Assumes Linearity**: Relationship may not always be linear.
- **Sensitive to Outliers**: Outliers can heavily influence the model.
- **Limited Flexibility**: May not capture complex patterns.

---

## **Naive Bayes Algorithm** 🧠

### **Introduction**
- **Definition**: A classification technique based on Bayes' Theorem with an assumption of feature independence.
- **Based on**: Bayes' Theorem.

### **Mind Map**
![Naive Bayes Mind Map](https://github.com/ParimalA24-DS/MACHINE-LEARNING-ALGORITHM-MINDMAP/blob/main/MLMINDMAPS/3.NAIVEBAYES_MINDMAP.PNG))

### **Types of Naive Bayes**
- **Gaussian Naive Bayes**: 
  - **Use Case**: Continuous data, e.g., predicting height based on age and weight.
- **Multinomial Naive Bayes**: 
  - **Use Case**: Text classification, e.g., classifying emails as spam or not spam based on word counts.
- **Bernoulli Naive Bayes**: 
  - **Use Case**: Binary features, e.g., predicting if a customer will buy a product based on page visits.

### **Assumptions**
- **Feature Independence**: Assumes features are independent of each other.
- **Equal Contribution**: Assumes each feature contributes equally to the outcome.

### **Formula**
- **Bayes' Theorem**: 
  - Formula: \( P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \)
  - **Explanation**: Calculates the probability of a class given the features.

### **Steps Involved**
1. **Convert the dataset into frequency tables**: Count occurrences of features for each class.
2. **Create likelihood tables**: Compute probabilities for each feature given a class.
3. **Calculate posterior probabilities**: Use Bayes' Theorem to compute the probability of each class.
4. **Classify based on highest posterior probability**: Choose the class with the highest probability.

### **Applications**
- **Spam Filtering**: Identifying whether an email is spam or not.
- **Sentiment Analysis**: Determining the sentiment of a text, e.g., positive or negative.
- **Medical Diagnosis**: Predicting the likelihood of a disease based on symptoms.

### **Advantages**
- **Simple and Fast**: Easy to implement and quick to train.
- **Handles Missing Data**: Can handle missing values well.
- **Performs Well in High-Dimensional Spaces**: Effective with many features.

### **Disadvantages**
- **Assumption of Feature Independence**: The assumption may not hold true for all datasets.
- **Zero Probability Issue**: If a feature value was not seen during training, it can lead to zero probability.

---

# References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Coursera Machine Learning Course](https://www.coursera.org/learn/machine-learning)

