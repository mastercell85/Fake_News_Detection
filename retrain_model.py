"""
Quick script to retrain the model with current scikit-learn version
"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("Loading training data...")
# Load the training data
train_news = pd.read_csv('train.csv')

# Handle missing values
train_news = train_news.fillna(' ')

# Create the pipeline with the best parameters (from classifier.py line 171-174)
print("Creating and training the model...")
logR_pipeline_ngram = Pipeline([
    ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,5), use_idf=True, smooth_idf=False)),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1, max_iter=1000))
])

# Train the model
logR_pipeline_ngram.fit(train_news['Statement'], train_news['Label'])

# Save the model
print("Saving the model to final_model.sav...")
model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram, open(model_file, 'wb'))

print("Model retrained and saved successfully!")
print("You can now run prediction.py")