# train.py
# Training script for Fake News Detection using LIAR dataset
# Adds dynamic speaker credibility as an additional feature

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack  # to combine sparse matrices
from utils import compute_speaker_scores, get_speaker_score_dynamic

print("✅ Starting train.py")

# Step 1: Load LIAR dataset (TSV) without headers
df = pd.read_csv('liar_dataset/train.tsv', sep='\t', header=None)

# Step 2: Inspect first row and number of columns
print("Number of columns detected:", len(df.columns))
print("First row sample:", df.iloc[0].tolist())

# Step 3: Assign column names dynamically based on number of columns
# Default LIAR column names (15 columns)
default_cols = [
    'id','label','statement','subject','speaker','job','state','party',
    'barely-true','false','half-true','mostly-true','true','pants-fire','context'
]

# Keep only as many names as the file has columns
df.columns = default_cols[:len(df.columns)]

# Step 4: Keep only the columns we need
# If your dataset does not have 'speaker', create a placeholder
if 'speaker' not in df.columns:
    df['speaker'] = 'Unknown'

df = df[['statement', 'label', 'speaker']]
df.rename(columns={'statement':'text'}, inplace=True)

# Step 5: Map original LIAR labels to binary
true_labels = ['true', 'mostly-true', 'half-true']
df['label'] = df['label'].apply(lambda x: 1 if x in true_labels else 0)

# Step 6: Fill missing text
df['text'] = df['text'].fillna("")

print(f"✅ Loaded dataset with {len(df)} rows")

# Step 7: Compute dynamic speaker credibility scores
speaker_scores = compute_speaker_scores(df, speaker_col='speaker', label_col='label')
print("✅ Speaker scores computed")

# Step 8: Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
text_features = vectorizer.fit_transform(df['text'])
print("✅ TF-IDF complete")

# Step 9: Generate speaker credibility feature
speaker_features = df['speaker'].apply(
    lambda x: get_speaker_score_dynamic(x, speaker_scores)
).values.reshape(-1, 1)

# Step 10: Combine text features with speaker credibility
X = hstack([text_features, speaker_features])
y = df['label']

# Step 11: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 12: Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("✅ Model trained")

# Step 13: Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.4f}")
