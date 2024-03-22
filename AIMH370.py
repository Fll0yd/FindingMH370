import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import joblib
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize logging
logging.basicConfig(level=logging.INFO)

def visualize_confusion_matrix(conf_matrix, title):
    """Visualize the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(fpr, tpr, label, title):
    """Plot the ROC curve."""
    plt.plot(fpr, tpr, linestyle='-', label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the model and print evaluation metrics."""
    predictions = model.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
    auc_roc = roc_auc_score(y_test, predictions, average='weighted', multi_class='ovr')

    logging.info(f"{model_name} Evaluation:")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1_score:.4f}")
    logging.info(f"AUC-ROC: {auc_roc:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    visualize_confusion_matrix(conf_matrix, f'Confusion Matrix - {model_name}')

    # ROC Curve
    probs = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs, pos_label=model.classes_[1])
    plot_roc_curve(fpr, tpr, f'ROC Curve - {model_name}', f'ROC Curve - {model_name}')

def load_data_from_mongodb(connection_string, database_name, collection_name):
    """Load data from MongoDB collection."""
    try:
        client = MongoClient(connection_string)
        db = client[database_name]
        collection = db[collection_name]
        data = list(collection.find())
        return data
    except Exception as e:
        logging.error(f"Error occurred while loading data from MongoDB: {e}")
        return None
    
def preprocess_text(text):
    """Preprocess text data."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def preprocess_data(data):
    """Preprocess the entire dataset."""
    if not data:
        return None, None
    X_text = [preprocess_text(d.get('snippet', '')) for d in data]  # Added .get() to handle missing keys
    X_meta = [d.get('metadata', '') for d in data]  # Added .get() to handle missing keys
    X_combined = [' '.join([X_text[i], X_meta[i]]) for i in range(len(X_text))]
    y = [d.get('title', '') for d in data]  # Added .get() to handle missing keys
    return X_combined, y

def main(args):
    """Main function to run the program."""
    # Load scraped data from MongoDB
    scraped_data = load_data_from_mongodb(args.connection_string, args.database_name, args.collection_name)

    if scraped_data:
        # Preprocess data
        X, y = preprocess_data(scraped_data)

        if X and y:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

            # SVM Classifier with TF-IDF Features
            svm_model = make_pipeline(
                TfidfVectorizer(),
                SVC(kernel='linear')
            )
            # Define parameter grid for GridSearchCV
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto']
            }
            # Initialize GridSearchCV
            grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            # Train the model with GridSearchCV
            grid_search.fit(X_train, y_train)
            # Get the best model
            best_svm_model = grid_search.best_estimator_
            # Save the best model
            joblib.dump(best_svm_model, 'best_svm_model.joblib')
            # Evaluate the best SVM model
            evaluate_model(best_svm_model, X_test, y_test, 'Best SVM with TF-IDF Features')
        else:
            logging.error("Error: Failed to preprocess data.")
    else:
        logging.error("Error: No data loaded. Check the MongoDB connection or the collection name.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Classification with SVM and BERT')
    parser.add_argument('--connection_string', type=str, default='mongodb://localhost:27017/', help='MongoDB connection string')
    parser.add_argument('--database_name', type=str, default='your_database', help='MongoDB database name')
    parser.add_argument('--collection_name', type=str, default='your_collection', help='MongoDB collection name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (fraction of total data)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train-test split')
    args = parser.parse_args()
    main(args)
