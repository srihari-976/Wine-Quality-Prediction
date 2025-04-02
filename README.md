# Wine Quality Predictor

A machine learning application that predicts the quality of red wine based on its physicochemical properties. The model uses an optimized Random Forest Classifier to predict wine quality on a scale of 0-8.

## Features

- Predicts wine quality based on 11 physicochemical properties
- Simple and intuitive user interface
- Shows feature importance for predictions
- Real-time quality assessment
- Optimized model using GridSearchCV
- Feature importance visualization

## Model Performance

The model achieves an accuracy of approximately 66.56% on the test set, with the following performance metrics:
- Best cross-validation score: 69.28%
- Best parameters: 
  - n_estimators: 300
  - max_depth: 20
  - min_samples_split: 5
  - min_samples_leaf: 1

## Example Cases

### Average Quality (5/8)
```
Fixed Acidity: 7.4
Volatile Acidity: 0.7
Citric Acid: 0.0
Residual Sugar: 1.9
Chlorides: 0.076
Free Sulfur Dioxide: 11.0
Total Sulfur Dioxide: 34.0
Density: 0.9978
pH: 3.51
Sulphates: 0.56
Alcohol: 9.4
```

### Good Quality (6/8)
```
Fixed Acidity: 7.8
Volatile Acidity: 0.88
Citric Acid: 0.0
Residual Sugar: 2.6
Chlorides: 0.098
Free Sulfur Dioxide: 25.0
Total Sulfur Dioxide: 67.0
Density: 0.9968
pH: 3.20
Sulphates: 0.68
Alcohol: 9.8
```

### Very Good Quality (7/8)
```
Fixed Acidity: 7.3
Volatile Acidity: 0.65
Citric Acid: 0.0
Residual Sugar: 1.2
Chlorides: 0.065
Free Sulfur Dioxide: 15.0
Total Sulfur Dioxide: 21.0
Density: 0.9946
pH: 3.39
Sulphates: 0.47
Alcohol: 10.0
```

### Excellent Quality (8/8)
```
Fixed Acidity: 7.5
Volatile Acidity: 0.5
Citric Acid: 0.36
Residual Sugar: 6.1
Chlorides: 0.071
Free Sulfur Dioxide: 17.0
Total Sulfur Dioxide: 102.0
Density: 0.9978
pH: 3.35
Sulphates: 0.8
Alcohol: 10.5
```

### Poor Quality (3/8)
```
Fixed Acidity: 7.2
Volatile Acidity: 0.8
Citric Acid: 0.0
Residual Sugar: 2.0
Chlorides: 0.097
Free Sulfur Dioxide: 9.0
Total Sulfur Dioxide: 18.0
Density: 0.9968
pH: 3.42
Sulphates: 0.44
Alcohol: 9.0
```

## Input Ranges

- Fixed Acidity: 4.0 - 16.0
- Volatile Acidity: 0.0 - 2.0
- Citric Acid: 0.0 - 1.0
- Residual Sugar: 0.0 - 16.0
- Chlorides: 0.0 - 1.0
- Free Sulfur Dioxide: 0 - 100
- Total Sulfur Dioxide: 0 - 300
- Density: 0.9 - 1.1
- pH: 2.0 - 4.0
- Sulphates: 0.0 - 2.0
- Alcohol: 8.0 - 15.0

## Quality Interpretation

- 3: Poor
- 4: Fair
- 5: Average
- 6: Good
- 7: Very Good
- 8: Excellent

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Run the training script:
```bash
python train.py
```
4. Start the Streamlit app:
```bash
streamlit run app.py
```

## Model Details

The model uses a Random Forest Classifier with the following optimizations:
- Grid Search Cross-Validation for hyperparameter tuning
- StandardScaler for feature scaling
- Feature importance analysis
- Performance metrics including accuracy, precision, recall, and F1-score

The model's performance varies across different quality levels:
- Best performance on average quality wines (5/8)
- Moderate performance on good (6/8) and very good (7/8) wines
- Limited performance on extreme quality levels (3/8, 4/8, 8/8) due to class imbalance 