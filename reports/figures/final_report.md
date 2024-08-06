# Final Report: Student Performance Analysis

## Overview

This report provides an overview of the student performance prediction project, highlighting the methods used, the models trained, and the insights gained from the analysis.

## Data Preprocessing

### Data Collection
- **Source:** The dataset was collected from Sling Academy.
- **Features:** The dataset contains various features including student demographics, academic performance, and extracurricular activities.

### Preprocessing Steps
1. **Handling Missing Values:** Imputation was done using the mean for numerical columns.
2. **Feature Scaling:** MinMaxScaler was applied to numerical features for normalization.
3. **One-Hot Encoding:** Categorical variables, such as career aspirations and gender, were one-hot encoded to facilitate model training.

## Model Training and Evaluation

### Models Trained
1. **Support Vector Machine (SVM):**
   - **Cross-Validation Accuracy:** 69.88%
   - **Test Accuracy:** 68.75%
   - **Confusion Matrix:** ![SVM Confusion Matrix](path_to_image)

2. **Logistic Regression:**
   - **Cross-Validation Accuracy:** 96.75%
   - **Test Accuracy:** 96.50%
   - **Confusion Matrix:** ![Logistic Regression Confusion Matrix](path_to_image)

3. **Decision Tree:**
   - **Cross-Validation Accuracy:** 80.38%
   - **Test Accuracy:** 82.25%
   - **Confusion Matrix:** ![Decision Tree Confusion Matrix](path_to_image)

4. **Neural Network:**
   - **Test Accuracy:** 77.25%
   - **Confusion Matrix:** ![Neural Network Confusion Matrix](path_to_image)

### Results and Insights
- Logistic Regression performed the best with the highest accuracy, both during cross-validation and on the test set.
- The Neural Network model, despite its complexity, did not outperform simpler models like Logistic Regression and Decision Tree.
- **Key Insight:** The high accuracy of the Logistic Regression model suggests that the relationships between features and student performance are mostly linear.

## Future Work

- **Neural Network Improvement:** Consider fine-tuning the neural network architecture or adding more training data to improve its performance.
- **Feature Engineering:** Further feature engineering could help capture more nuanced relationships in the data.
- **Additional Models:** Experimenting with ensemble methods such as Random Forests or Gradient Boosting could provide more insights.

## Conclusion

The project successfully demonstrated the ability to predict student performance using various machine learning models. The Logistic Regression model provided the best balance of simplicity and accuracy, making it a strong candidate for practical applications in educational settings.
