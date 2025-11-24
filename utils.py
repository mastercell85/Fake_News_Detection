# utils.py
# Utility functions for speaker credibility score

import pandas as pd

def compute_speaker_scores(df, speaker_col='speaker', label_col='label'):
    """
    Computes credibility scores for each speaker based on historical article labels.
    
    Parameters:
    - df: pandas DataFrame containing the news dataset
    - speaker_col: column name containing speaker/author names
    - label_col: column name containing article labels (1=real, 0=fake)
    
    Returns:
    - speaker_scores: dictionary {speaker_name: credibility_score}
    """
    # Group by speaker and calculate mean label (fraction of real articles) and count of articles
    speaker_counts = df.groupby(speaker_col)[label_col].agg(['mean', 'count'])
    
    # Apply smoothing to avoid extreme scores for speakers with very few articles
    smoothing = 2  # pseudo-count
    # Compute smoothed credibility score: blends speaker's mean with neutral 0.5
    speaker_counts['score'] = (
        (speaker_counts['mean'] * speaker_counts['count'] + 0.5 * smoothing) 
        / (speaker_counts['count'] + smoothing)
    )
    
    # Convert the result to a dictionary: {speaker_name: credibility_score}
    speaker_scores = speaker_counts['score'].to_dict()
    return speaker_scores

def get_speaker_score_dynamic(speaker_name, speaker_scores):
    """
    Fetches the credibility score for a given speaker.
    
    Parameters:
    - speaker_name: name of the speaker/author
    - speaker_scores: dictionary from compute_speaker_scores
    
    Returns:
    - credibility score (0 to 1)
    - defaults to 0.5 if speaker is unknown
    """
    return speaker_scores.get(speaker_name, 0.5)
