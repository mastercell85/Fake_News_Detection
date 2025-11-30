# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:45:40 2017

@author: NishitP
Updated: 2025-01-15 - Added keyword highlighting feature
Updated: 2025-01-15 - Added model comparison tracker (old vs enhanced)
Updated: 2025-01-15 - Added speaker credibility feature (from utils.py)
Updated: 2025-01-15 - Added Google Fact Check API integration
Updated: 2025-01-28 - Added security: input validation & secure API key loading
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import os

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import security module for input validation
try:
    from security import validate_and_sanitize_input, InputValidator, secure_config
    SECURITY_ENABLED = True
    print("Security Module: Enabled (input validation active)")
except ImportError:
    SECURITY_ENABLED = False
    print("Security Module: Not available")

# Import fact-check API module
try:
    from factcheck_api import get_fact_check_summary, check_api_configured, display_fact_check_results
    FACTCHECK_AVAILABLE = check_api_configured()
    if FACTCHECK_AVAILABLE:
        print("Google Fact Check API: Enabled")
    else:
        print("Google Fact Check API: Not configured (set API key in .env)")
except ImportError:
    FACTCHECK_AVAILABLE = False
    print("Google Fact Check API: Module not found")

# Load model once at startup (more efficient)
print("Loading model...")
load_model = pickle.load(open('final_model.sav', 'rb'))
print("Model loaded successfully!")

# ============================================================================
# SPEAKER CREDIBILITY SYSTEM (from utils.py)
# ============================================================================

def compute_speaker_scores(df, speaker_col='speaker', label_col='label'):
    """
    Computes credibility scores for each speaker based on historical article labels.

    How it works:
    1. Groups all statements by speaker
    2. Calculates the percentage of TRUE statements for each speaker
    3. Applies smoothing to avoid extreme scores for speakers with few statements

    Formula: score = (mean * count + 0.5 * smoothing) / (count + smoothing)
    - mean = fraction of TRUE statements (0 to 1)
    - count = number of statements by this speaker
    - smoothing = 2 (pseudo-count to push toward 0.5 for unknown speakers)

    Parameters:
    - df: pandas DataFrame containing the news dataset
    - speaker_col: column name containing speaker/author names
    - label_col: column name containing article labels (1=real, 0=fake)

    Returns:
    - speaker_scores: dictionary {speaker_name: credibility_score}
    """
    # Group by speaker and calculate mean label (fraction of real articles) and count
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


def get_speaker_score(speaker_name, speaker_scores):
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


def load_speaker_credibility_data():
    """
    Load and compute speaker credibility scores from the LIAR dataset.

    The LIAR dataset contains speaker information that we use to calculate
    historical credibility scores based on past truthfulness.

    Returns:
    - speaker_scores: dictionary {speaker_name: credibility_score}
    - speaker_info: dictionary {speaker_name: {job_title, party, state, counts}}
    """
    # Try to load from the LIAR dataset TSV files
    liar_train_path = 'liar_dataset/train.tsv'

    if not os.path.exists(liar_train_path):
        print("Warning: LIAR dataset not found. Speaker credibility disabled.")
        return {}, {}

    # LIAR dataset columns (TSV with no header):
    # 0: ID, 1: label, 2: statement, 3: subject, 4: speaker, 5: job_title,
    # 6: state, 7: party, 8-12: credit history counts, 13: context
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
               'state', 'party', 'barely_true', 'false', 'half_true',
               'mostly_true', 'pants_on_fire', 'context']

    try:
        df = pd.read_csv(liar_train_path, sep='\t', header=None, names=columns)

        # Clean up: remove rows with missing speaker names
        df = df.dropna(subset=['speaker'])
        df = df[df['speaker'].str.strip() != '']

        # Convert labels to binary (1 = true-ish, 0 = false-ish)
        # true, mostly-true, half-true -> 1
        # barely-true, false, pants-on-fire -> 0
        true_labels = ['true', 'mostly-true', 'half-true']
        df['binary_label'] = df['label'].apply(lambda x: 1 if x in true_labels else 0)

        # Compute speaker scores
        speaker_scores = compute_speaker_scores(df, speaker_col='speaker', label_col='binary_label')

        # Also collect speaker info (job, party, state)
        speaker_info = {}
        for speaker in df['speaker'].unique():
            speaker_rows = df[df['speaker'] == speaker]
            if len(speaker_rows) > 0:
                speaker_data = speaker_rows.iloc[0]
                speaker_info[speaker] = {
                    'job_title': str(speaker_data.get('job_title', 'Unknown')) if pd.notna(speaker_data.get('job_title')) else 'Unknown',
                    'party': str(speaker_data.get('party', 'Unknown')) if pd.notna(speaker_data.get('party')) else 'Unknown',
                    'state': str(speaker_data.get('state', 'Unknown')) if pd.notna(speaker_data.get('state')) else 'Unknown',
                    'total_statements': len(speaker_rows)
                }

        print(f"Loaded speaker credibility data for {len(speaker_scores)} speakers.")
        return speaker_scores, speaker_info

    except Exception as e:
        print(f"Error loading speaker data: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}


# Load speaker credibility data at startup
print("Loading speaker credibility data...")
SPEAKER_SCORES, SPEAKER_INFO = load_speaker_credibility_data()
print("")


# ============================================================================
# BASELINE MODEL (Old/Original Prediction - Simple output)
# ============================================================================

def baseline_prediction(statement):
    """
    Original/baseline prediction function.

    This is the OLD model behavior - simple prediction without enhancements.
    Used as a comparison baseline to measure improvements.

    What it does:
    1. Takes the statement as input
    2. Runs it through the ML model
    3. Returns just the prediction and raw probability

    No keyword analysis, no explanations - just raw prediction.

    Args:
        statement (str): The news text to analyze

    Returns:
        dict: Contains prediction, probability, and method used
    """
    # Make prediction using the base model
    prediction = load_model.predict([statement])[0]
    prob = load_model.predict_proba([statement])

    # Get raw probabilities for both classes
    prob_false = prob[0][0] * 100
    prob_true = prob[0][1] * 100

    # Confidence is the probability of the predicted class
    # Note: prediction is numpy.bool (True/False), not string
    if prediction:  # True
        confidence = prob_true
    else:  # False
        confidence = prob_false

    return {
        'method': 'BASELINE (Old Model)',
        'prediction': prediction,
        'confidence': confidence,
        'prob_true': prob_true,
        'prob_false': prob_false,
        'has_keywords': False,
        'has_explanation': False
    }


# ============================================================================
# ENHANCED MODEL (New Prediction - With keyword analysis)
# ============================================================================

def enhanced_prediction(statement, speaker=None, use_factcheck=True):
    """
    Enhanced prediction function with keyword analysis, speaker credibility, and fact-check API.

    This is the NEW model behavior - includes all enhancements:
    - Keyword highlighting
    - Impact analysis
    - Explanation generation
    - Speaker credibility scoring
    - Google Fact Check API integration (NEW)

    Args:
        statement (str): The news text to analyze
        speaker (str): Optional speaker/author name for credibility lookup
        use_factcheck (bool): Whether to query Google Fact Check API (default: True)

    Returns:
        dict: Contains prediction, probability, keywords, explanation, speaker info, and fact-check data
    """
    # Make prediction using the base model
    prediction = load_model.predict([statement])[0]
    prob = load_model.predict_proba([statement])

    # Get raw probabilities for both classes
    prob_false = prob[0][0] * 100
    prob_true = prob[0][1] * 100

    # Base ML confidence (before speaker adjustment)
    # Note: prediction is numpy.bool (True/False), not string 'True'/'False'
    if prediction:  # True
        ml_confidence = prob_true
    else:  # False
        ml_confidence = prob_false

    # Get influential keywords
    influential_words = get_influential_words(statement, load_model)

    # Generate explanation (will be updated later if prediction is flipped)
    explanation = generate_explanation(prediction, influential_words)
    original_explanation = explanation  # Keep original for reference

    # Calculate enhancement score (how much keyword analysis adds)
    enhancement_score = calculate_enhancement_score(influential_words)

    # ========================================
    # SPEAKER CREDIBILITY INTEGRATION
    # ========================================
    speaker_data = None
    speaker_score = 0.5  # Default neutral score
    adjusted_confidence = ml_confidence
    speaker_adjustment = 0
    final_prediction = prediction  # May be changed by speaker credibility

    if speaker and SPEAKER_SCORES:
        # Normalize speaker name for matching
        # Handle formats like "Donald Trump" -> "donald-trump"
        speaker_normalized = speaker.lower().strip().replace(' ', '-')

        # Try to find speaker (case-insensitive search with multiple formats)
        matched_speaker = None
        for s in SPEAKER_SCORES.keys():
            s_lower = s.lower()
            # Match exact, with spaces, or with hyphens
            if s_lower == speaker_normalized or s_lower == speaker.lower().strip():
                matched_speaker = s
                break

        if matched_speaker:
            speaker_score = SPEAKER_SCORES[matched_speaker]
            speaker_info_data = SPEAKER_INFO.get(matched_speaker, {})

            speaker_data = {
                'name': matched_speaker,
                'credibility_score': speaker_score,
                'job_title': speaker_info_data.get('job_title', 'Unknown'),
                'party': speaker_info_data.get('party', 'Unknown'),
                'state': speaker_info_data.get('state', 'Unknown'),
                'total_statements': speaker_info_data.get('total_statements', 0),
                'found': True
            }

            # ========================================
            # ENHANCED SPEAKER CREDIBILITY LOGIC
            # ========================================
            # The speaker's credibility can now FLIP the prediction if:
            # 1. The ML model is uncertain (confidence < 70%)
            # 2. AND the speaker has a strong credibility signal (very high or very low)
            #
            # This makes the model smarter by using speaker history.

            speaker_weight = 25  # Increased weight for stronger impact

            # Calculate how much to adjust based on speaker vs neutral (0.5)
            credibility_deviation = speaker_score - 0.5  # Negative = less credible, Positive = more credible

            if prediction:  # ML says TRUE (numpy.bool True, not string)
                # ML says TRUE. Check if speaker credibility disagrees.
                # Low credibility speaker + TRUE prediction = reduce confidence significantly
                speaker_adjustment = credibility_deviation * speaker_weight * 2

                # FLIP LOGIC: If speaker is very untrustworthy and ML isn't confident
                if speaker_score < 0.35 and ml_confidence < 70:
                    # Flip to FALSE - this speaker is historically unreliable
                    final_prediction = False  # Use boolean, not string
                    # New confidence based on how untrustworthy + original uncertainty
                    adjusted_confidence = min(100, (0.5 - speaker_score) * 100 + (100 - ml_confidence) * 0.5)
                    speaker_data['prediction_flipped'] = True
                    speaker_data['flip_reason'] = f"Speaker credibility ({speaker_score:.2f}) is very low and ML was uncertain ({ml_confidence:.1f}%)"
                    # Update explanation to reflect the flip
                    explanation = f"Predicted FALSE because speaker '{matched_speaker}' has very low credibility (0.27) based on historical fact-checks."
                else:
                    adjusted_confidence = min(100, max(0, ml_confidence + speaker_adjustment))
            else:  # ML says FALSE (numpy.bool False)
                # ML says FALSE. Check if speaker credibility disagrees.
                # High credibility speaker + FALSE prediction = reduce confidence
                speaker_adjustment = (0.5 - speaker_score) * speaker_weight * 2

                # FLIP LOGIC: If speaker is very trustworthy and ML isn't confident
                if speaker_score > 0.65 and ml_confidence < 70:
                    # Flip to TRUE - this speaker is historically reliable
                    final_prediction = True  # Use boolean, not string
                    adjusted_confidence = min(100, (speaker_score - 0.5) * 100 + (100 - ml_confidence) * 0.5)
                    speaker_data['prediction_flipped'] = True
                    speaker_data['flip_reason'] = f"Speaker credibility ({speaker_score:.2f}) is high and ML was uncertain ({ml_confidence:.1f}%)"
                    # Update explanation to reflect the flip
                    explanation = f"Predicted TRUE because speaker '{matched_speaker}' has high credibility ({speaker_score:.2f}) based on historical fact-checks."
                else:
                    adjusted_confidence = min(100, max(0, ml_confidence + speaker_adjustment))
        else:
            speaker_data = {
                'name': speaker,
                'credibility_score': 0.5,
                'found': False,
                'message': 'Speaker not found in database. Using neutral score.'
            }

    # ========================================
    # GOOGLE FACT CHECK API INTEGRATION
    # ========================================
    factcheck_data = None
    factcheck_adjustment = 0

    if use_factcheck and FACTCHECK_AVAILABLE:
        # Query Google Fact Check API
        factcheck_data = get_fact_check_summary(statement)

        if factcheck_data.get('available', False):
            # Get the confidence modifier from fact-check results
            factcheck_adjustment = factcheck_data.get('confidence_modifier', 0)
            verdict = factcheck_data.get('verdict', 'UNKNOWN')

            # Apply fact-check adjustment to confidence
            # Fact-check is authoritative, so it can significantly affect the result
            if verdict == 'FALSE':
                # Fact-checkers say FALSE - if we predicted TRUE, flip it
                if final_prediction:  # Currently predicting TRUE
                    # Strong evidence it's false - flip prediction
                    final_prediction = False
                    adjusted_confidence = min(100, max(50, 70 + abs(factcheck_adjustment) * 0.5))
                    factcheck_data['prediction_flipped'] = True
                    factcheck_data['flip_reason'] = f"Professional fact-checkers rate this claim as FALSE"
                    # Update explanation to reflect fact-check override
                    explanation = f"Predicted FALSE because professional fact-checkers (PolitiFact, Snopes, etc.) have rated this claim as false."
                else:
                    # Already predicting FALSE - boost confidence
                    adjusted_confidence = min(100, adjusted_confidence + abs(factcheck_adjustment))

            elif verdict == 'TRUE':
                # Fact-checkers say TRUE - if we predicted FALSE, flip it
                if not final_prediction:  # Currently predicting FALSE
                    # Strong evidence it's true - flip prediction
                    final_prediction = True
                    adjusted_confidence = min(100, max(50, 70 + abs(factcheck_adjustment) * 0.5))
                    factcheck_data['prediction_flipped'] = True
                    factcheck_data['flip_reason'] = f"Professional fact-checkers rate this claim as TRUE"
                    # Update explanation to reflect fact-check override
                    explanation = f"Predicted TRUE because professional fact-checkers (PolitiFact, Snopes, etc.) have verified this claim as true."
                else:
                    # Already predicting TRUE - boost confidence
                    adjusted_confidence = min(100, adjusted_confidence + abs(factcheck_adjustment))

            # For MIXED or INCONCLUSIVE, don't change prediction, just note it

    return {
        'method': 'ENHANCED (New Model)',
        'prediction': final_prediction,  # May differ from ML prediction if speaker/factcheck flipped it
        'ml_prediction': prediction,      # Original ML prediction (before adjustments)
        'ml_confidence': ml_confidence,
        'confidence': adjusted_confidence,  # Final confidence (after all adjustments)
        'prob_true': prob_true,
        'prob_false': prob_false,
        'has_keywords': True,
        'has_explanation': True,
        'has_speaker_credibility': speaker_data is not None,
        'has_factcheck': factcheck_data is not None and factcheck_data.get('available', False),
        'keywords': influential_words,
        'explanation': explanation,
        'original_ml_explanation': original_explanation,  # What ML alone would have said
        'enhancement_score': enhancement_score,
        'speaker': speaker_data,
        'speaker_adjustment': speaker_adjustment,
        'factcheck': factcheck_data,
        'factcheck_adjustment': factcheck_adjustment
    }


def generate_explanation(prediction, influential_words):
    """
    Generate a human-readable explanation of the prediction.

    Args:
        prediction: The True/False prediction
        influential_words: Dictionary of words pushing True/False

    Returns:
        str: Plain English explanation
    """
    # Note: prediction can be numpy.bool or Python bool (True/False), not string
    if not prediction:  # False
        if influential_words['pushing_false']:
            top_word = influential_words['pushing_false'][0]['word']
            return f"Predicted FALSE due to words like \"{top_word}\" associated with misleading statements."
        else:
            return "Predicted FALSE based on overall text pattern."
    else:  # True
        if influential_words['pushing_true']:
            top_word = influential_words['pushing_true'][0]['word']
            return f"Predicted TRUE due to words like \"{top_word}\" associated with factual statements."
        else:
            return "Predicted TRUE based on overall text pattern."


def calculate_enhancement_score(influential_words):
    """
    Calculate a score showing how much the keyword analysis enhances understanding.

    Higher score = more informative keyword analysis

    How it works:
    1. Count significant keywords found (impact > threshold)
    2. Calculate total absolute impact
    3. Check if both True and False indicators are present (nuanced)

    Args:
        influential_words: Dictionary of keywords and their impacts

    Returns:
        dict: Enhancement metrics
    """
    all_words = influential_words.get('all_words', [])
    true_words = influential_words.get('pushing_true', [])
    false_words = influential_words.get('pushing_false', [])

    # Count significant keywords (impact > 0.1)
    significant_count = len([w for w in all_words if abs(w['impact']) > 0.1])

    # Total absolute impact (sum of all impacts)
    total_impact = sum(abs(w['impact']) for w in all_words)

    # Is the analysis nuanced? (has both True and False indicators)
    is_nuanced = len(true_words) > 0 and len(false_words) > 0

    # Calculate enhancement score (0-100)
    # Higher = more informative keyword analysis
    score = min(100, (significant_count * 10) + (total_impact * 20))

    return {
        'score': score,
        'significant_keywords': significant_count,
        'total_impact': total_impact,
        'is_nuanced': is_nuanced,
        'true_indicators': len(true_words),
        'false_indicators': len(false_words)
    }


# ============================================================================
# COMPARISON TRACKER
# ============================================================================

def run_comparison(statement, speaker=None):
    """
    Run both baseline and enhanced models, then compare results.

    This allows you to see:
    1. What the OLD model would have shown (NO speaker credibility)
    2. What the NEW model shows (WITH speaker credibility + keyword analysis)
    3. Whether predictions match
    4. How much value the enhancements add

    Args:
        statement (str): The news text to analyze
        speaker (str): Optional speaker name for credibility lookup (enhanced model only)

    Returns:
        dict: Comparison results from both models
    """
    # Run baseline (old) model - NO speaker credibility
    baseline = baseline_prediction(statement)

    # Run enhanced (new) model - WITH speaker credibility if provided
    enhanced = enhanced_prediction(statement, speaker=speaker)

    # Compare results
    predictions_match = baseline['prediction'] == enhanced['prediction']
    confidence_diff = enhanced['confidence'] - baseline['confidence']

    return {
        'statement': statement,
        'speaker': speaker,
        'baseline': baseline,
        'enhanced': enhanced,
        'predictions_match': predictions_match,
        'confidence_difference': confidence_diff
    }


def display_comparison(comparison):
    """
    Display side-by-side comparison of baseline vs enhanced results.

    Shows:
    1. Both predictions
    2. Confidence scores
    3. Whether they agree
    4. What the enhancements add
    5. Speaker credibility information (if provided)
    """
    baseline = comparison['baseline']
    enhanced = comparison['enhanced']

    print("\n" + "="*70)
    print("MODEL COMPARISON: BASELINE vs ENHANCED")
    print("="*70)

    print(f"\nStatement: {comparison['statement']}")

    # Show speaker if provided
    if comparison.get('speaker'):
        print(f"Speaker: {comparison['speaker']}")

    # Side-by-side comparison box
    print("\n" + "-"*70)
    print(f"{'BASELINE (Old Model)':<35} | {'ENHANCED (New Model)':<35}")
    print("-"*70)
    # Convert predictions to readable True/False strings
    baseline_pred = "True" if baseline['prediction'] else "False"
    enhanced_pred = "True" if enhanced['prediction'] else "False"
    print(f"Prediction: {baseline_pred:<23} | Prediction: {enhanced_pred:<23}")
    print(f"Confidence: {baseline['confidence']:.2f}%{'':<17} | Confidence: {enhanced['confidence']:.2f}%")
    print(f"Keywords:   {'No':<23} | Keywords:   {'Yes':<23}")
    print(f"Explanation: {'No':<22} | Explanation: {'Yes':<22}")

    # Show speaker credibility status
    has_speaker = enhanced.get('has_speaker_credibility', False)
    has_factcheck = enhanced.get('has_factcheck', False)
    print(f"Speaker Cred: {'No':<21} | Speaker Cred: {'Yes' if has_speaker else 'No':<21}")
    print(f"Fact Check:   {'No':<21} | Fact Check:   {'Yes' if has_factcheck else 'No':<21}")
    print("-"*70)

    # Agreement status
    if comparison['predictions_match']:
        print("\n[✓] PREDICTIONS MATCH - Both models agree")
    else:
        print("\n[!] PREDICTIONS DIFFER - Models disagree!")
        print(f"    Baseline says: {baseline['prediction']}")
        print(f"    Enhanced says: {enhanced['prediction']}")

    # Show if prediction was FLIPPED by speaker credibility
    if enhanced.get('speaker') and enhanced['speaker'].get('prediction_flipped'):
        print("\n" + "!"*70)
        print("[PREDICTION FLIPPED BY SPEAKER CREDIBILITY]")
        print("!"*70)
        print(f"    ML Model predicted: {enhanced.get('ml_prediction', 'Unknown')}")
        print(f"    Final prediction:   {enhanced['prediction']}")
        print(f"    Reason: {enhanced['speaker'].get('flip_reason', 'Speaker history override')}")

    # Show confidence difference if speaker affected it
    elif enhanced.get('speaker_adjustment', 0) != 0:
        ml_conf = enhanced.get('ml_confidence', enhanced['confidence'])
        print(f"\n[i] SPEAKER CREDIBILITY IMPACT:")
        print(f"    ML-only confidence: {ml_conf:.2f}%")
        print(f"    Speaker adjustment: {enhanced['speaker_adjustment']:+.2f}%")
        print(f"    Final confidence:   {enhanced['confidence']:.2f}%")

    # ========================================
    # SPEAKER CREDIBILITY SECTION
    # ========================================
    if enhanced.get('speaker'):
        speaker_data = enhanced['speaker']
        print("\n" + "-"*70)
        print("SPEAKER CREDIBILITY ANALYSIS:")
        print("-"*70)

        if speaker_data.get('found', False):
            # Display speaker info
            cred_score = speaker_data['credibility_score']
            cred_bar = "█" * int(cred_score * 20)
            empty_bar = "░" * (20 - int(cred_score * 20))

            print(f"  Speaker: {speaker_data['name']}")
            print(f"  Credibility Score: {cred_bar}{empty_bar} {cred_score:.2f}")

            # Interpret credibility
            if cred_score >= 0.7:
                cred_level = "HIGH (historically truthful)"
            elif cred_score >= 0.5:
                cred_level = "MODERATE (mixed record)"
            elif cred_score >= 0.3:
                cred_level = "LOW (often misleading)"
            else:
                cred_level = "VERY LOW (frequently false)"
            print(f"  Credibility Level: {cred_level}")

            # Show additional speaker info if available
            if speaker_data.get('job_title') and speaker_data['job_title'] != 'Unknown':
                print(f"  Job Title: {speaker_data['job_title']}")
            if speaker_data.get('party') and speaker_data['party'] != 'Unknown':
                print(f"  Party: {speaker_data['party']}")
            if speaker_data.get('state') and speaker_data['state'] != 'Unknown':
                print(f"  State: {speaker_data['state']}")
            if speaker_data.get('total_statements', 0) > 0:
                print(f"  Statements in Database: {speaker_data['total_statements']}")
        else:
            print(f"  Speaker: {speaker_data.get('name', 'Unknown')}")
            print(f"  Status: NOT FOUND in database")
            print(f"  Using neutral credibility score: 0.50")

    # ========================================
    # FACT CHECK API SECTION
    # ========================================
    if enhanced.get('factcheck'):
        factcheck_data = enhanced['factcheck']
        print("\n" + "-"*70)
        print("GOOGLE FACT CHECK API RESULTS:")
        print("-"*70)

        if factcheck_data.get('available', False):
            verdict = factcheck_data.get('verdict', 'UNKNOWN')

            # Display verdict with visual indicator
            if verdict == 'FALSE':
                print(f"  Verdict: [X] FALSE - Fact-checkers rate this claim as false")
            elif verdict == 'TRUE':
                print(f"  Verdict: [✓] TRUE - Fact-checkers rate this claim as true")
            elif verdict == 'MIXED':
                print(f"  Verdict: [~] MIXED - Fact-checkers have mixed ratings")
            else:
                print(f"  Verdict: [?] {verdict} - Unable to determine clear verdict")

            # Show if prediction was flipped by fact-check
            if factcheck_data.get('prediction_flipped'):
                print(f"\n  ** PREDICTION OVERRIDDEN BY FACT-CHECK **")
                print(f"  Reason: {factcheck_data.get('flip_reason', 'Fact-check evidence')}")

            # Show rating breakdown
            breakdown = factcheck_data.get('rating_breakdown', {})
            if breakdown:
                print(f"\n  Rating Breakdown:")
                print(f"    False ratings: {breakdown.get('false', 0)}")
                print(f"    True ratings:  {breakdown.get('true', 0)}")
                print(f"    Mixed ratings: {breakdown.get('mixed', 0)}")

            # Show confidence impact
            modifier = enhanced.get('factcheck_adjustment', 0)
            if modifier != 0:
                print(f"\n  Confidence Impact: {modifier:+.1f}%")

            # Show sources
            sources = factcheck_data.get('sources', [])
            if sources:
                print(f"\n  Sources ({len(sources)} fact-checks found):")
                for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                    print(f"    {i}. {source['publisher']}: {source['rating']}")
                    if source.get('url'):
                        # Truncate long URLs
                        url = source['url']
                        if len(url) > 50:
                            url = url[:50] + "..."
                        print(f"       {url}")
        else:
            if factcheck_data.get('error'):
                print(f"  Error: {factcheck_data['error']}")
            else:
                print("  No fact-checks found for this claim.")
                print("  (The claim may not have been reviewed by fact-checkers yet)")

    # Enhancement value
    print("\n" + "-"*70)
    print("KEYWORD ANALYSIS:")
    print("-"*70)

    enh_score = enhanced['enhancement_score']
    print(f"  Enhancement Score: {enh_score['score']:.1f}/100")
    print(f"  Significant Keywords Found: {enh_score['significant_keywords']}")
    print(f"  Total Keyword Impact: {enh_score['total_impact']:.4f}")
    print(f"  Nuanced Analysis: {'Yes (both True & False indicators)' if enh_score['is_nuanced'] else 'No (one-sided)'}")

    # Show keywords if available
    if enhanced['keywords']['pushing_false']:
        print(f"\n  Top False Indicators:")
        for i, word in enumerate(enhanced['keywords']['pushing_false'][:3], 1):
            print(f"    {i}. \"{word['word']}\" (impact: {abs(word['impact']):.4f})")

    if enhanced['keywords']['pushing_true']:
        print(f"\n  Top True Indicators:")
        for i, word in enumerate(enhanced['keywords']['pushing_true'][:3], 1):
            print(f"    {i}. \"{word['word']}\" (impact: {word['impact']:.4f})")

    # Explanation
    print(f"\n  Explanation: {enhanced['explanation']}")

    # Show final summary if prediction was modified
    if enhanced.get('speaker') and enhanced['speaker'].get('prediction_flipped'):
        print("\n" + "-"*70)
        print("FINAL VERDICT:")
        print("-"*70)
        final_pred = "FALSE (Fake News)" if not enhanced['prediction'] else "TRUE (Real News)"
        print(f"  {final_pred}")
        print(f"  Reason: Speaker's low credibility overrode uncertain ML prediction")
    elif enhanced.get('factcheck') and enhanced['factcheck'].get('prediction_flipped'):
        print("\n" + "-"*70)
        print("FINAL VERDICT:")
        print("-"*70)
        final_pred = "FALSE (Fake News)" if not enhanced['prediction'] else "TRUE (Real News)"
        print(f"  {final_pred}")
        print(f"  Reason: Professional fact-checkers overrode ML prediction")

    print("="*70 + "\n")


def get_influential_words(statement, model, top_n=10):
    """
    Extract the most influential words that affected the prediction.

    How it works:
    1. The TfidfVectorizer converts words to numerical features
    2. The LogisticRegression has coefficients (weights) for each feature
    3. Words with HIGH POSITIVE coefficients push toward "True"
    4. Words with HIGH NEGATIVE coefficients push toward "False"
    5. We multiply each word's TF-IDF score by its coefficient to get impact

    Args:
        statement (str): The input text to analyze
        model: The trained pipeline (TfidfVectorizer + LogisticRegression)
        top_n (int): Number of top influential words to return

    Returns:
        dict: Contains words pushing toward True and False predictions
    """
    # Get the vectorizer and classifier from the pipeline
    vectorizer = model.named_steps['LogR_tfidf']
    classifier = model.named_steps['LogR_clf']

    # Transform the statement to TF-IDF features
    tfidf_matrix = vectorizer.transform([statement])

    # Get feature names (all possible words/n-grams the model knows)
    feature_names = vectorizer.get_feature_names_out()

    # Get the classifier's coefficients (how much each feature influences prediction)
    # Positive coefficient = pushes toward "True"
    # Negative coefficient = pushes toward "False"
    coefficients = classifier.coef_[0]

    # Get the TF-IDF values for this specific statement
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Calculate impact: TF-IDF score * coefficient
    # This shows how much each word actually influenced THIS prediction
    impacts = tfidf_scores * coefficients

    # Find words that are actually in the statement (non-zero TF-IDF)
    non_zero_indices = np.where(tfidf_scores > 0)[0]

    # Collect influential words with their impacts
    word_impacts = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        impact = impacts[idx]
        tfidf = tfidf_scores[idx]
        coef = coefficients[idx]
        word_impacts.append({
            'word': word,
            'impact': impact,
            'tfidf': tfidf,
            'coefficient': coef
        })

    # Sort by absolute impact (most influential first)
    word_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)

    # Separate into words pushing True vs False
    true_words = [w for w in word_impacts if w['impact'] > 0][:top_n]
    false_words = [w for w in word_impacts if w['impact'] < 0][:top_n]

    return {
        'pushing_true': true_words,
        'pushing_false': false_words,
        'all_words': word_impacts[:top_n]
    }


def display_highlighted_results(statement, prediction, probability, influential_words):
    """
    Display the prediction results with keyword highlighting.

    Shows which words pushed the prediction toward True or False.
    """
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION RESULT")
    print("="*70)

    print(f"\nStatement: {statement}")
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {probability:.2f}%")

    # Show words pushing toward FALSE (likely fake news indicators)
    print("\n" + "-"*70)
    print("KEYWORDS PUSHING TOWARD 'FALSE' (Fake News Indicators):")
    print("-"*70)
    if influential_words['pushing_false']:
        for i, word_info in enumerate(influential_words['pushing_false'][:5], 1):
            impact_bar = "█" * min(int(abs(word_info['impact']) * 50), 20)
            print(f"  {i}. \"{word_info['word']}\"")
            print(f"     Impact: {impact_bar} ({abs(word_info['impact']):.4f})")
    else:
        print("  No strong fake news indicators found.")

    # Show words pushing toward TRUE (credibility indicators)
    print("\n" + "-"*70)
    print("KEYWORDS PUSHING TOWARD 'TRUE' (Credibility Indicators):")
    print("-"*70)
    if influential_words['pushing_true']:
        for i, word_info in enumerate(influential_words['pushing_true'][:5], 1):
            impact_bar = "█" * min(int(abs(word_info['impact']) * 50), 20)
            print(f"  {i}. \"{word_info['word']}\"")
            print(f"     Impact: {impact_bar} ({word_info['impact']:.4f})")
    else:
        print("  No strong credibility indicators found.")

    # Summary explanation
    print("\n" + "-"*70)
    print("EXPLANATION:")
    print("-"*70)
    # Note: prediction can be numpy.bool or Python bool
    if not prediction:  # False
        if influential_words['pushing_false']:
            top_false = influential_words['pushing_false'][0]['word']
            print(f"  The model predicted FALSE primarily because of words like \"{top_false}\"")
            print(f"  which are commonly associated with misleading statements in the training data.")
    else:  # True
        if influential_words['pushing_true']:
            top_true = influential_words['pushing_true'][0]['word']
            print(f"  The model predicted TRUE primarily because of words like \"{top_true}\"")
            print(f"  which are commonly associated with factual statements in the training data.")

    print("="*70 + "\n")


def detecting_fake_news(var):
    """
    Main prediction function with keyword highlighting.

    Analyzes the input statement and shows:
    1. True/False prediction
    2. Confidence score
    3. Keywords that influenced the prediction
    """
    # Validate input using security module if available
    if SECURITY_ENABLED:
        validation = InputValidator.validate_statement(var)
        if not validation['valid']:
            print(f"Error: {validation['error']}")
            return
        var = validation['sanitized']
    elif not var or not var.strip():
        print("Error: Please enter a valid statement.")
        return

    # Make prediction
    prediction = load_model.predict([var])[0]
    prob = load_model.predict_proba([var])

    # Get the confidence (probability of the predicted class)
    # Note: prediction is numpy.bool, not string
    if prediction:  # True
        probability = prob[0][1] * 100  # Probability of True
    else:  # False
        probability = prob[0][0] * 100  # Probability of False

    # Get influential words
    influential_words = get_influential_words(var, load_model)

    # Display results with highlighting
    display_highlighted_results(var, prediction, probability, influential_words)

    return prediction, probability, influential_words


if __name__ == '__main__':
    print("="*70)
    print("FAKE NEWS DETECTOR WITH MODEL COMPARISON")
    print("="*70)
    print("\nThis tool analyzes news statements and compares:")
    print("  - BASELINE: Original model (prediction + confidence only)")
    print("  - ENHANCED: New model (keywords + speaker credibility + explanations)")
    print("\nMODES:")
    print("  1 = Comparison Mode (shows both old and new model results)")
    print("  2 = Enhanced Only (new model with keywords + speaker)")
    print("  3 = Baseline Only (old model, simple output)")
    print("\nCOMMANDS:")
    print("  'quit'     - Exit the program")
    print("  'mode'     - Change operating mode")
    print("  'speakers' - List some known speakers in the database")
    print("")

    # Default to comparison mode
    current_mode = 1

    while True:
        var = input("Enter news statement to verify: ")

        if var.lower() == 'quit':
            print("Goodbye!")
            break

        if var.lower() == 'mode':
            print("\nSelect mode:")
            print("  1 = Comparison Mode (shows both)")
            print("  2 = Enhanced Only (with keywords + speaker)")
            print("  3 = Baseline Only (simple)")
            mode_input = input("Enter mode (1/2/3): ")
            if mode_input in ['1', '2', '3']:
                current_mode = int(mode_input)
                mode_names = {1: 'Comparison', 2: 'Enhanced Only', 3: 'Baseline Only'}
                print(f"Mode changed to: {mode_names[current_mode]}\n")
            else:
                print("Invalid mode. Keeping current mode.\n")
            continue

        if var.lower() == 'speakers':
            # Show some known speakers
            print("\n" + "-"*70)
            print("SAMPLE SPEAKERS IN DATABASE:")
            print("-"*70)
            if SPEAKER_SCORES:
                # Sort by credibility score and show top/bottom speakers
                sorted_speakers = sorted(SPEAKER_SCORES.items(), key=lambda x: x[1], reverse=True)
                print("\nMost Credible Speakers:")
                for name, score in sorted_speakers[:5]:
                    print(f"  - {name}: {score:.2f}")
                print("\nLeast Credible Speakers:")
                for name, score in sorted_speakers[-5:]:
                    print(f"  - {name}: {score:.2f}")
                print(f"\nTotal speakers in database: {len(SPEAKER_SCORES)}")
            else:
                print("No speaker data loaded.")
            print("-"*70 + "\n")
            continue

        if var.strip():
            # ========================================
            # SECURITY: Validate and sanitize input
            # ========================================
            if SECURITY_ENABLED:
                validation_result = validate_and_sanitize_input(var)
                if not validation_result['valid']:
                    print("\n[SECURITY] Input validation failed:")
                    for error in validation_result['errors']:
                        print(f"  - {error}")
                    print("Please try again with valid input.\n")
                    continue
                # Use sanitized input
                var = validation_result['statement']

            # Ask for speaker name (optional)
            speaker = None
            if current_mode in [1, 2]:  # Only ask for speaker in comparison/enhanced modes
                speaker_input = input("Enter speaker name (or press Enter to skip): ").strip()
                if speaker_input:
                    # Validate speaker name if security is enabled
                    if SECURITY_ENABLED:
                        speaker_validation = InputValidator.validate_speaker(speaker_input)
                        if not speaker_validation['valid']:
                            print(f"\n[SECURITY] Speaker name validation failed: {speaker_validation['error']}")
                            print("Continuing without speaker...\n")
                            speaker = None
                        else:
                            speaker = speaker_validation['sanitized'] if speaker_validation['sanitized'] else speaker_input
                    else:
                        speaker = speaker_input

            if current_mode == 1:
                # Comparison mode - show both models
                comparison = run_comparison(var, speaker=speaker)
                display_comparison(comparison)
            elif current_mode == 2:
                # Enhanced mode - new model with keywords + speaker
                result = enhanced_prediction(var, speaker=speaker)
                print("\n" + "="*70)
                print("ENHANCED PREDICTION (New Model)")
                print("="*70)
                print(f"\nStatement: {var}")
                if speaker:
                    print(f"Speaker: {speaker}")
                print(f"\nPrediction: {result['prediction']}")
                if result.get('speaker_adjustment', 0) != 0:
                    print(f"ML Confidence: {result['ml_confidence']:.2f}%")
                    print(f"Speaker Adjustment: {result['speaker_adjustment']:+.2f}%")
                print(f"Final Confidence: {result['confidence']:.2f}%")

                # Show speaker info
                if result.get('speaker') and result['speaker'].get('found'):
                    spk = result['speaker']
                    print(f"\nSpeaker Credibility: {spk['credibility_score']:.2f}")
                    if spk.get('job_title') and spk['job_title'] != 'Unknown':
                        print(f"Job Title: {spk['job_title']}")

                # Show fact-check results
                if result.get('factcheck'):
                    fc = result['factcheck']
                    print("\n" + "-"*70)
                    print("FACT CHECK RESULTS:")
                    if fc.get('available'):
                        verdict = fc.get('verdict', 'UNKNOWN')
                        print(f"  Verdict: {verdict}")
                        if fc.get('prediction_flipped'):
                            print(f"  ** Prediction was overridden by fact-check **")
                        sources = fc.get('sources', [])
                        if sources:
                            print(f"  Sources: {', '.join([s['publisher'] for s in sources[:3]])}")
                    else:
                        print("  No fact-checks found for this claim.")

                # Show keywords
                print("\n" + "-"*70)
                print("TOP KEYWORDS:")
                if result['keywords']['pushing_false']:
                    print("  False Indicators:", ", ".join([f"\"{w['word']}\"" for w in result['keywords']['pushing_false'][:3]]))
                if result['keywords']['pushing_true']:
                    print("  True Indicators:", ", ".join([f"\"{w['word']}\"" for w in result['keywords']['pushing_true'][:3]]))

                print(f"\nExplanation: {result['explanation']}")
                print("="*70 + "\n")
            else:
                # Baseline mode - old simple output (NO speaker)
                result = baseline_prediction(var)
                print("\n" + "="*70)
                print("BASELINE PREDICTION (Old Model)")
                print("="*70)
                print(f"\nStatement: {var}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                print("="*70 + "\n")
        else:
            print("Please enter a valid statement.\n")