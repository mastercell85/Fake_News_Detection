# security.py
# Security Module for Fake News Detection System
#
# This module provides:
# 1. Secure environment variable loading with validation
# 2. Input validation and sanitization
# 3. Rate limiting helpers
#
# SECURITY IMPLEMENTATION #1: Environment Variables
# SECURITY IMPLEMENTATION #2: Input Validation & Sanitization

import os
import re
import html
from pathlib import Path


# ============================================================================
# SECURITY #1: SECURE ENVIRONMENT VARIABLE LOADING
# ============================================================================

class SecureConfig:
    """
    Secure configuration loader for API keys and sensitive data.

    Security features:
    - Loads from .env file (never hardcoded)
    - Validates API key format
    - Prevents logging of sensitive values
    - Checks file permissions (advisory)
    """

    _instance = None
    _api_key = None
    _is_loaded = False

    def __new__(cls):
        """Singleton pattern to ensure only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, env_path='.env'):
        """
        Securely load configuration from .env file.

        Args:
            env_path: Path to .env file (default: '.env')

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if self._is_loaded:
            return True

        env_file = Path(env_path)

        # Check if .env file exists
        if not env_file.exists():
            print("[SECURITY] Warning: .env file not found. API features disabled.")
            return False

        # Advisory: Check file permissions (Windows doesn't have same permission model)
        try:
            # On Unix-like systems, warn if file is world-readable
            if hasattr(os, 'stat'):
                import stat
                mode = os.stat(env_path).st_mode
                if mode & stat.S_IROTH:  # World-readable
                    print("[SECURITY] Warning: .env file may be readable by other users.")
        except Exception:
            pass  # Permission check is advisory only

        # Parse .env file manually (more secure than exec-based loaders)
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # Store in environment
                        os.environ[key] = value

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"[SECURITY] Error loading .env file: {e}")
            return False

    def get_api_key(self, key_name='GOOGLE_FACTCHECK_API_KEY'):
        """
        Securely retrieve API key with validation.

        Args:
            key_name: Environment variable name for the API key

        Returns:
            str or None: The API key if valid, None otherwise
        """
        if not self._is_loaded:
            self.load_config()

        api_key = os.environ.get(key_name)

        # Validate the API key
        if not api_key:
            return None

        if api_key == 'YOUR_API_KEY_HERE':
            return None

        # Basic format validation for Google API keys
        # Google API keys are typically 39 characters, alphanumeric with some special chars
        if not self._validate_api_key_format(api_key):
            print(f"[SECURITY] Warning: API key format appears invalid.")
            # Still return it - might be a different format we don't know about

        return api_key

    def _validate_api_key_format(self, api_key):
        """
        Validate API key format (basic checks).

        Google API keys are typically:
        - 39 characters long
        - Start with 'AIza'
        - Contain alphanumeric chars and some special chars

        Returns:
            bool: True if format looks valid
        """
        if not api_key:
            return False

        # Check minimum length
        if len(api_key) < 20:
            return False

        # Check for common placeholder values
        placeholder_patterns = [
            'your_key', 'api_key', 'xxx', 'test', 'example',
            '1234', 'abcd', 'placeholder'
        ]
        api_key_lower = api_key.lower()
        for pattern in placeholder_patterns:
            if pattern in api_key_lower:
                return False

        return True

    def mask_api_key(self, api_key):
        """
        Mask API key for safe logging/display.

        Args:
            api_key: The API key to mask

        Returns:
            str: Masked version like "AIza****...****Xyz"
        """
        if not api_key or len(api_key) < 8:
            return "****"
        return f"{api_key[:4]}****...****{api_key[-3:]}"


# Global config instance
secure_config = SecureConfig()


# ============================================================================
# SECURITY #2: INPUT VALIDATION & SANITIZATION
# ============================================================================

class InputValidator:
    """
    Input validation and sanitization for user inputs.

    Protects against:
    - Injection attacks (SQL, command, etc.)
    - XSS (Cross-Site Scripting)
    - Oversized inputs (DoS)
    - Malicious Unicode/encoding
    """

    # Configuration
    MAX_STATEMENT_LENGTH = 5000  # Maximum characters for a news statement
    MAX_SPEAKER_LENGTH = 200    # Maximum characters for speaker name
    MIN_STATEMENT_LENGTH = 3    # Minimum characters for meaningful analysis

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>',           # Script tags
        r'javascript:',              # JavaScript URLs
        r'on\w+\s*=',               # Event handlers (onclick, onload, etc.)
        r'data:text/html',          # Data URLs with HTML
        r'\x00',                     # Null bytes
        r'&#x?[0-9a-f]+;',          # Encoded characters that might be malicious
    ]

    # Compile patterns for efficiency
    _dangerous_regex = None

    @classmethod
    def _get_dangerous_regex(cls):
        """Compile dangerous patterns regex (cached)."""
        if cls._dangerous_regex is None:
            pattern = '|'.join(cls.DANGEROUS_PATTERNS)
            cls._dangerous_regex = re.compile(pattern, re.IGNORECASE)
        return cls._dangerous_regex

    @classmethod
    def validate_statement(cls, statement):
        """
        Validate and sanitize a news statement input.

        Args:
            statement: The user-provided statement to validate

        Returns:
            dict: {
                'valid': bool,
                'sanitized': str or None,
                'error': str or None
            }
        """
        # Check if input exists
        if statement is None:
            return {
                'valid': False,
                'sanitized': None,
                'error': 'Statement cannot be empty.'
            }

        # Convert to string if needed
        if not isinstance(statement, str):
            try:
                statement = str(statement)
            except Exception:
                return {
                    'valid': False,
                    'sanitized': None,
                    'error': 'Invalid input type.'
                }

        # Strip whitespace
        statement = statement.strip()

        # Check minimum length
        if len(statement) < cls.MIN_STATEMENT_LENGTH:
            return {
                'valid': False,
                'sanitized': None,
                'error': f'Statement too short. Minimum {cls.MIN_STATEMENT_LENGTH} characters required.'
            }

        # Check maximum length (prevent DoS)
        if len(statement) > cls.MAX_STATEMENT_LENGTH:
            return {
                'valid': False,
                'sanitized': None,
                'error': f'Statement too long. Maximum {cls.MAX_STATEMENT_LENGTH} characters allowed.'
            }

        # Check for dangerous patterns
        dangerous_regex = cls._get_dangerous_regex()
        if dangerous_regex.search(statement):
            return {
                'valid': False,
                'sanitized': None,
                'error': 'Input contains potentially unsafe content.'
            }

        # Sanitize the input
        sanitized = cls._sanitize_text(statement)

        return {
            'valid': True,
            'sanitized': sanitized,
            'error': None
        }

    @classmethod
    def validate_speaker(cls, speaker):
        """
        Validate and sanitize a speaker name input.

        Args:
            speaker: The user-provided speaker name

        Returns:
            dict: {
                'valid': bool,
                'sanitized': str or None,
                'error': str or None
            }
        """
        # Speaker is optional - empty is valid
        if speaker is None or (isinstance(speaker, str) and speaker.strip() == ''):
            return {
                'valid': True,
                'sanitized': None,
                'error': None
            }

        # Convert to string if needed
        if not isinstance(speaker, str):
            try:
                speaker = str(speaker)
            except Exception:
                return {
                    'valid': False,
                    'sanitized': None,
                    'error': 'Invalid speaker name type.'
                }

        # Strip whitespace
        speaker = speaker.strip()

        # Check maximum length
        if len(speaker) > cls.MAX_SPEAKER_LENGTH:
            return {
                'valid': False,
                'sanitized': None,
                'error': f'Speaker name too long. Maximum {cls.MAX_SPEAKER_LENGTH} characters.'
            }

        # Check for dangerous patterns
        dangerous_regex = cls._get_dangerous_regex()
        if dangerous_regex.search(speaker):
            return {
                'valid': False,
                'sanitized': None,
                'error': 'Speaker name contains invalid characters.'
            }

        # Sanitize - speaker names should only have letters, spaces, hyphens, periods
        # Allow international characters but sanitize potentially dangerous ones
        sanitized = cls._sanitize_speaker_name(speaker)

        return {
            'valid': True,
            'sanitized': sanitized,
            'error': None
        }

    @classmethod
    def _sanitize_text(cls, text):
        """
        Sanitize text input by escaping/removing dangerous content.

        Args:
            text: The text to sanitize

        Returns:
            str: Sanitized text
        """
        # HTML escape to prevent XSS if output is ever rendered as HTML
        # (defensive measure even though we're CLI)
        sanitized = html.escape(text)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Normalize whitespace (collapse multiple spaces/newlines)
        sanitized = ' '.join(sanitized.split())

        return sanitized

    @classmethod
    def _sanitize_speaker_name(cls, name):
        """
        Sanitize speaker name - more restrictive than general text.

        Args:
            name: The speaker name to sanitize

        Returns:
            str: Sanitized speaker name
        """
        # HTML escape
        sanitized = html.escape(name)

        # Remove characters that aren't typically in names
        # Allow: letters (including international), spaces, hyphens, periods, apostrophes
        sanitized = re.sub(r'[^\w\s\-\.\'\,]', '', sanitized, flags=re.UNICODE)

        # Collapse whitespace
        sanitized = ' '.join(sanitized.split())

        return sanitized

    @classmethod
    def sanitize_for_api(cls, text):
        """
        Sanitize text specifically for sending to external APIs.

        More aggressive sanitization for external communication.

        Args:
            text: Text to sanitize for API use

        Returns:
            str: API-safe text
        """
        if not text:
            return ''

        # First do standard sanitization
        sanitized = cls._sanitize_text(text)

        # Additional API-specific sanitization
        # Remove any remaining HTML entities (send plain text to API)
        sanitized = html.unescape(sanitized)

        # Limit length for API calls
        max_api_length = 1000
        if len(sanitized) > max_api_length:
            sanitized = sanitized[:max_api_length]

        return sanitized


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_and_sanitize_input(statement, speaker=None):
    """
    Convenience function to validate both statement and speaker.

    Args:
        statement: The news statement to analyze
        speaker: Optional speaker name

    Returns:
        dict: {
            'valid': bool,
            'statement': sanitized statement or None,
            'speaker': sanitized speaker or None,
            'errors': list of error messages
        }
    """
    errors = []

    # Validate statement
    stmt_result = InputValidator.validate_statement(statement)
    if not stmt_result['valid']:
        errors.append(stmt_result['error'])

    # Validate speaker
    spkr_result = InputValidator.validate_speaker(speaker)
    if not spkr_result['valid']:
        errors.append(spkr_result['error'])

    return {
        'valid': len(errors) == 0,
        'statement': stmt_result['sanitized'] if stmt_result['valid'] else None,
        'speaker': spkr_result['sanitized'],
        'errors': errors
    }


def get_secure_api_key():
    """
    Get the API key securely.

    Returns:
        str or None: The API key if configured, None otherwise
    """
    return secure_config.get_api_key()


def is_api_configured():
    """
    Check if API is properly configured.

    Returns:
        bool: True if API key is available and valid
    """
    return get_secure_api_key() is not None


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("SECURITY MODULE TEST")
    print("="*70)

    # Test 1: Environment variable loading
    print("\n1. Testing secure config loading...")
    secure_config.load_config()
    api_key = secure_config.get_api_key()
    if api_key:
        print(f"   API Key found: {secure_config.mask_api_key(api_key)}")
    else:
        print("   API Key not configured (this is OK for testing)")

    # Test 2: Input validation
    print("\n2. Testing input validation...")

    test_cases = [
        ("Normal statement about politics", True),
        ("", False),  # Empty
        ("ab", False),  # Too short
        ("<script>alert('xss')</script>", False),  # XSS attempt
        ("A" * 6000, False),  # Too long
        ("Normal claim with numbers 123", True),
        ("Statement with 'quotes' and \"double quotes\"", True),
    ]

    for statement, expected_valid in test_cases:
        result = InputValidator.validate_statement(statement)
        status = "✓" if result['valid'] == expected_valid else "✗"
        display = statement[:40] + "..." if len(statement) > 40 else statement
        print(f"   {status} '{display}' -> valid={result['valid']}")

    # Test 3: Speaker validation
    print("\n3. Testing speaker validation...")

    speaker_tests = [
        ("Donald Trump", True),
        ("", True),  # Empty is OK (optional)
        (None, True),  # None is OK (optional)
        ("A" * 250, False),  # Too long
        ("<script>bad</script>", False),
    ]

    for speaker, expected_valid in speaker_tests:
        result = InputValidator.validate_speaker(speaker)
        status = "✓" if result['valid'] == expected_valid else "✗"
        display = str(speaker)[:30] if speaker else "None"
        print(f"   {status} '{display}' -> valid={result['valid']}")

    print("\n" + "="*70)
    print("Security module test complete!")