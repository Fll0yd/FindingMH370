import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Function to load data from JSON files
def load_data_from_json(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                json_data = json.load(file)
                data.append(json_data)
    return data

# Function to preprocess data and split it into features and labels
def preprocess_data(data):
    X = [d['snippet'] for d in data]  # Snippet as feature
    y = [d['title'] for d in data]    # Title as label
    return X, y

# Load scraped data from specified drive
data_directory = "H:/MH370_data"
scraped_data = load_data_from_json(data_directory)

# Preprocess data
X, y = preprocess_data(scraped_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline for text classification
model = make_pipeline(
    CountVectorizer(),  # Convert text to vectors
    MultinomialNB()     # Naive Bayes classifier
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
