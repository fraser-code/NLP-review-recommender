# NLP-review-recommender

# Predicting Customer Recommendations for StyleSense

## Getting Started

This project aims to predict whether a customer would recommend a product based on their review text, demographics, and product details. The goal is to enhance customer insights and improve inventory and marketing decisions for StyleSense, an online women's clothing retailer.

### Dependencies

Ensure you have Python 3 installed. Install the required libraries using:

```
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install spacy
!pip install imbalanced-learn
!pip install sentence-transformers
!pip install joblib
```

### Installation

Follow these steps to set up the environment:

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd Prediction-Model
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```
   jupyter notebook
   ```

## Testing

To validate the model, follow these steps:

1. Run the notebook cells sequentially.
2. Evaluate model performance using:
   ```
   from sklearn.metrics import classification_report
   print(classification_report(y_test, y_pred))
   ```
3. Analyze confusion matrices and accuracy scores.

### Break Down Tests

- **Data Preprocessing**: Ensures missing values are handled and categorical data is encoded.
- **Model Training**: Uses Logistic Regression and other classifiers to optimize performance.
- **Evaluation**: Confusion matrix, classification report, and accuracy metrics are used to assess model predictions.

## Project Instructions

- Load and preprocess the dataset.
- Train and evaluate predictive models.
- Interpret results and refine the model.

## Built With

- [scikit-learn](https://scikit-learn.org/) - Machine Learning Library
- [pandas](https://pandas.pydata.org/) - Data Handling
- [NumPy](https://numpy.org/) - Numerical Computing
- [imblearn](https://imbalanced-learn.org/) - Handling Imbalanced Datasets
- [sentence-transformers](https://www.sbert.net/) - Text Embeddings

## License

This project follows the [Udacity License](https://github.com/udacity/dsnd-pipelines-project/blob/main/LICENSE.txt).

