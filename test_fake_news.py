import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def compute_speaker_scores(df, speaker_col='speaker', label_col='label'):
    speaker_counts = df.groupby(speaker_col)[label_col].agg(['mean', 'count'])
    smoothing = 2  # pseudo-count to avoid extremes
    speaker_counts['score'] = ((speaker_counts['mean'] * speaker_counts['count'] + 0.5 * smoothing) 
                               / (speaker_counts['count'] + smoothing))
    return speaker_counts['score'].to_dict()

def get_speaker_score_dynamic(speaker_name, speaker_scores):
    return speaker_scores.get(speaker_name, 0.5)  # default 0.5 if unknown

# -----------------------
# SAMPLE DATASET
# -----------------------
data = {
    'text': [
        "Breaking news: Market hits record high",
        "Aliens landed in New York City",
        "New study shows coffee improves memory",
        "Chocolate cures all diseases",
        "Local team wins championship",
        "Government hiding the truth about UFOs",
        "Scientists discover new species of bird",
        "Miracle weight loss pills exposed"
    ],
    'label': [1,0,1,0,1,0,1,0],
    'speaker': ["Alice Smith","John Doe","Alice Smith","John Doe",
                "Bob Lee","Jane Roe","Bob Lee","Jane Roe"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# -----------------------
# COMPUTE SPEAKER CREDIBILITY
# -----------------------
speaker_scores = compute_speaker_scores(df)
df['speaker_score'] = df['speaker'].apply(lambda x: get_speaker_score_dynamic(x, speaker_scores))

# Print speaker scores for verification
print("Speaker Credibility Scores:")
for speaker, score in speaker_scores.items():
    print(f"{speaker}: {score:.2f}")

# -----------------------
# TEXT FEATURE EXTRACTION
# -----------------------
vectorizer = TfidfVectorizer(max_features=50)
text_features = vectorizer.fit_transform(df['text'])

# Combine text features with speaker credibility
speaker_features = df['speaker_score'].values.reshape(-1,1)
X = hstack([text_features, speaker_features])
y = df['label']

# -----------------------
# TRAIN TEST SPLIT AND MODEL
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# -----------------------
# EVALUATION
# -----------------------
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
