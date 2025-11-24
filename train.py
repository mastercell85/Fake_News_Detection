# train.py
# Training script for Fake News Detection
# Adds dynamic speaker credibility as an additional feature

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack  # to combine sparse matrices
from utils import compute_speaker_scores, get_speaker_score_dynamic

# Step 1: Load dataset
# Assume CSV has columns: 'text', 'label', 'speaker'
df = pd.read_csv('data/news_data.csv')

# Step 2: Compute speaker credibility scores dynamically
# This creates a dictionary {speaker_name: score} based on historical labels
speaker_scores = compute_speaker_scores(df, speaker_col='speaker', label_col='label')

# Step 3: Convert text into TF-IDF features
# max_features limits the number of features for efficiency
vectorizer = TfidfVectorizer(max_features=5000)
text_features = vectorizer.fit_transform(df['text'])

# Step 4: Generate speaker credibility feature
# Apply the dynamic score function to each row's speaker
speaker_features = df['speaker'].apply(
    lambda x: get_speaker_score_dynamic(x, speaker_scores)
).values.reshape(-1, 1)  # reshape to 2D array for stacking

# Step 5: Combine text features with speaker credibility
# hstack allows us to combine sparse text matrix with dense speaker feature
X = hstack([text_features, speaker_features])

# Labels (1=real, 0=fake)
y = df['label']

# Step 6: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train the classifier
# Using Random Forest; you can replace with other models if desired
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 8: Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
