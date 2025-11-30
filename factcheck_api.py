# factcheck_api.py
# Google Fact Check Tools API Integration
#
# This module provides functions to query the Google Fact Check Tools API
# to cross-reference claims with professional fact-checking organizations.
#
# SECURITY: Uses secure configuration loading from security.py

import requests

# Import security module for secure API key handling
try:
    from security import secure_config, is_api_configured, InputValidator
    SECURITY_MODULE_AVAILABLE = True
except ImportError:
    SECURITY_MODULE_AVAILABLE = False
    print("[SECURITY] Warning: security module not found. Using fallback.")
    import os
    from dotenv import load_dotenv
    load_dotenv()

# API Configuration
FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _get_api_key():
    """
    Get API key securely using the security module.

    Returns:
        str or None: The API key if configured, None otherwise
    """
    if SECURITY_MODULE_AVAILABLE:
        return secure_config.get_api_key('GOOGLE_FACTCHECK_API_KEY')
    else:
        # Fallback for when security module is not available
        import os
        key = os.getenv('GOOGLE_FACTCHECK_API_KEY')
        if key and key != 'YOUR_API_KEY_HERE':
            return key
        return None


def check_api_configured():
    """
    Check if the API key is properly configured.

    Returns:
        bool: True if API key is set, False otherwise
    """
    return _get_api_key() is not None


def search_fact_checks(query, language_code='en', max_results=5):
    """
    Search for fact-checks related to a claim using Google Fact Check Tools API.

    How it works:
    1. Sends the claim text to Google's Fact Check API
    2. Google searches across multiple fact-checking organizations
    3. Returns matching fact-checks with ratings and sources

    Parameters:
        query (str): The claim/statement to fact-check
        language_code (str): Language code for results (default: 'en')
        max_results (int): Maximum number of results to return (default: 5)

    Returns:
        dict: Contains:
            - success (bool): Whether the API call succeeded
            - found (bool): Whether any fact-checks were found
            - claims (list): List of fact-check results
            - error (str): Error message if failed
    """
    if not check_api_configured():
        return {
            'success': False,
            'found': False,
            'claims': [],
            'error': 'API key not configured. Please set GOOGLE_FACTCHECK_API_KEY in .env file.'
        }

    try:
        # Get API key securely
        api_key = _get_api_key()

        # Sanitize query for API call (if security module available)
        if SECURITY_MODULE_AVAILABLE:
            sanitized_query = InputValidator.sanitize_for_api(query)
        else:
            sanitized_query = query

        # Prepare API request parameters
        params = {
            'key': api_key,
            'query': sanitized_query,
            'languageCode': language_code,
            'pageSize': max_results
        }

        # Make the API request
        response = requests.get(FACTCHECK_API_URL, params=params, timeout=10)

        # Check for HTTP errors
        if response.status_code == 403:
            return {
                'success': False,
                'found': False,
                'claims': [],
                'error': 'API key invalid or Fact Check API not enabled. Check Google Cloud Console.'
            }
        elif response.status_code == 429:
            return {
                'success': False,
                'found': False,
                'claims': [],
                'error': 'Rate limit exceeded. Please wait and try again.'
            }
        elif response.status_code != 200:
            return {
                'success': False,
                'found': False,
                'claims': [],
                'error': f'API error: HTTP {response.status_code}'
            }

        # Parse the response
        data = response.json()

        # Check if any claims were found
        if 'claims' not in data or len(data['claims']) == 0:
            return {
                'success': True,
                'found': False,
                'claims': [],
                'message': 'No fact-checks found for this claim.'
            }

        # Process and structure the results
        processed_claims = []
        for claim in data['claims']:
            claim_info = {
                'text': claim.get('text', 'N/A'),
                'claimant': claim.get('claimant', 'Unknown'),
                'claim_date': claim.get('claimDate', 'Unknown'),
                'reviews': []
            }

            # Process claim reviews (fact-check ratings)
            for review in claim.get('claimReview', []):
                review_info = {
                    'publisher': review.get('publisher', {}).get('name', 'Unknown'),
                    'publisher_site': review.get('publisher', {}).get('site', ''),
                    'url': review.get('url', ''),
                    'title': review.get('title', ''),
                    'rating': review.get('textualRating', 'Unknown'),
                    'language': review.get('languageCode', 'en')
                }
                claim_info['reviews'].append(review_info)

            processed_claims.append(claim_info)

        return {
            'success': True,
            'found': True,
            'claims': processed_claims,
            'total_found': len(processed_claims)
        }

    except requests.exceptions.Timeout:
        return {
            'success': False,
            'found': False,
            'claims': [],
            'error': 'API request timed out. Please try again.'
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'found': False,
            'claims': [],
            'error': 'Network connection error. Check your internet connection.'
        }
    except Exception as e:
        return {
            'success': False,
            'found': False,
            'claims': [],
            'error': f'Unexpected error: {str(e)}'
        }


def get_fact_check_summary(query):
    """
    Get a summarized fact-check result for integration with the prediction model.

    This function is designed to be called by the enhanced_prediction() function
    to add fact-check data to the prediction results.

    Parameters:
        query (str): The claim/statement to fact-check

    Returns:
        dict: Structured summary containing:
            - available (bool): Whether fact-check data is available
            - verdict (str): Overall fact-check verdict (if found)
            - confidence_modifier (float): How much to adjust ML confidence (-50 to +50)
            - sources (list): List of fact-checking sources
            - details (dict): Full API response data
    """
    result = search_fact_checks(query)

    if not result['success']:
        return {
            'available': False,
            'error': result.get('error', 'Unknown error'),
            'verdict': None,
            'confidence_modifier': 0,
            'sources': [],
            'details': result
        }

    if not result['found']:
        return {
            'available': False,
            'message': 'No fact-checks found for this claim',
            'verdict': None,
            'confidence_modifier': 0,
            'sources': [],
            'details': result
        }

    # Analyze the fact-check results to determine overall verdict
    ratings = []
    sources = []

    for claim in result['claims']:
        for review in claim['reviews']:
            rating_text = review['rating'].lower()
            sources.append({
                'publisher': review['publisher'],
                'rating': review['rating'],
                'url': review['url']
            })

            # Categorize the rating
            # Common ratings: "False", "Mostly False", "Half True", "Mostly True", "True"
            # Also: "Pants on Fire", "Misleading", "Unproven", etc.
            if any(word in rating_text for word in ['false', 'pants on fire', 'misleading', 'incorrect', 'wrong', 'fake', 'fabricated']):
                ratings.append('FALSE')
            elif any(word in rating_text for word in ['true', 'correct', 'accurate', 'verified']):
                if 'mostly' in rating_text or 'partially' in rating_text:
                    ratings.append('MOSTLY_TRUE')
                else:
                    ratings.append('TRUE')
            elif any(word in rating_text for word in ['half', 'mixed', 'partly']):
                ratings.append('MIXED')
            elif any(word in rating_text for word in ['unproven', 'unverified', 'unknown']):
                ratings.append('UNVERIFIED')
            else:
                # Default categorization based on common patterns
                ratings.append('UNKNOWN')

    # Determine overall verdict and confidence modifier
    if not ratings:
        return {
            'available': True,
            'verdict': 'INCONCLUSIVE',
            'confidence_modifier': 0,
            'sources': sources,
            'details': result
        }

    # Count verdicts
    false_count = ratings.count('FALSE')
    true_count = ratings.count('TRUE') + ratings.count('MOSTLY_TRUE')
    mixed_count = ratings.count('MIXED')

    total = len(ratings)

    # Determine overall verdict
    if false_count > total / 2:
        verdict = 'FALSE'
        # Strong confidence that claim is false (decrease "True" confidence)
        confidence_modifier = -30 - (false_count / total * 20)  # -30 to -50
    elif true_count > total / 2:
        verdict = 'TRUE'
        # Strong confidence that claim is true (increase "True" confidence)
        confidence_modifier = 30 + (true_count / total * 20)  # +30 to +50
    elif mixed_count > 0:
        verdict = 'MIXED'
        confidence_modifier = 0  # No adjustment for mixed verdicts
    else:
        verdict = 'INCONCLUSIVE'
        confidence_modifier = 0

    return {
        'available': True,
        'verdict': verdict,
        'confidence_modifier': confidence_modifier,
        'sources': sources,
        'claim_count': len(result['claims']),
        'rating_breakdown': {
            'false': false_count,
            'true': true_count,
            'mixed': mixed_count,
            'total': total
        },
        'details': result
    }


def display_fact_check_results(fact_check_data):
    """
    Display fact-check results in a formatted way.

    Parameters:
        fact_check_data (dict): Result from get_fact_check_summary()
    """
    print("\n" + "-"*70)
    print("GOOGLE FACT CHECK API RESULTS:")
    print("-"*70)

    if not fact_check_data.get('available', False):
        if fact_check_data.get('error'):
            print(f"  Error: {fact_check_data['error']}")
        else:
            print("  No fact-checks found for this claim.")
            print("  (The claim may not have been reviewed by fact-checkers yet)")
        return

    verdict = fact_check_data.get('verdict', 'UNKNOWN')

    # Display verdict with visual indicator
    if verdict == 'FALSE':
        print(f"  Verdict: [X] FALSE - Fact-checkers rate this claim as false")
    elif verdict == 'TRUE':
        print(f"  Verdict: [✓] TRUE - Fact-checkers rate this claim as true")
    elif verdict == 'MIXED':
        print(f"  Verdict: [~] MIXED - Fact-checkers have mixed ratings")
    else:
        print(f"  Verdict: [?] INCONCLUSIVE - Unable to determine clear verdict")

    # Show rating breakdown
    breakdown = fact_check_data.get('rating_breakdown', {})
    if breakdown:
        print(f"\n  Rating Breakdown:")
        print(f"    False ratings: {breakdown.get('false', 0)}")
        print(f"    True ratings:  {breakdown.get('true', 0)}")
        print(f"    Mixed ratings: {breakdown.get('mixed', 0)}")

    # Show confidence modifier
    modifier = fact_check_data.get('confidence_modifier', 0)
    if modifier != 0:
        print(f"\n  Confidence Impact: {modifier:+.1f}%")

    # Show sources
    sources = fact_check_data.get('sources', [])
    if sources:
        print(f"\n  Sources ({len(sources)} fact-checks found):")
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            print(f"    {i}. {source['publisher']}: {source['rating']}")
            if source.get('url'):
                print(f"       URL: {source['url'][:60]}...")


# Test function
if __name__ == '__main__':
    print("="*70)
    print("GOOGLE FACT CHECK API TEST")
    print("="*70)

    if not check_api_configured():
        print("\nError: API key not configured!")
        print("Please edit the .env file and add your Google Fact Check API key.")
        print("\nSteps:")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create or select a project")
        print("3. Enable 'Fact Check Tools API'")
        print("4. Create an API key")
        print("5. Add it to .env file: GOOGLE_FACTCHECK_API_KEY=your_key_here")
    else:
        print("\nAPI key configured. Testing with sample claim...")

        # Test with a well-known fact-checked claim
        test_claim = "Mexico will pay for the wall"
        print(f"\nTest claim: \"{test_claim}\"")

        result = get_fact_check_summary(test_claim)
        display_fact_check_results(result)

        print("\n" + "="*70)
        print("Test complete!")