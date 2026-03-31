"""
Standalone test script for the updated TwitterScraper.
Imports only what's needed to avoid dependency issues.
"""

import asyncio
import json
import os
import sys

# Add the scrapers folder to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src/data/scrapers"))

# Now import just what we need
from loguru import logger
from tqdm import tqdm
from twikit import Client as TwikitClient

# Read the config from a simple JSON file
CONFIG_FILE = "scraper_config.json"

def load_config():
    """Load scraper configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "cookies_file": "src/data/scrapers/twitter_cookies.json",
        "language": "en-US"
    }

async def test_scraper():
    """Test the scraper with cookies."""
    config = load_config()
    
    cookies_file = config.get("cookies_file", "src/data/scrapers/twitter_cookies.json")
    
    if not os.path.exists(cookies_file):
        print(f"Cookies file not found: {cookies_file}")
        return
    
    print(f"Loading cookies from {cookies_file}...")
    
    # Load cookies
    with open(cookies_file, 'r') as f:
        browser_cookies = json.load(f)
    
    # Convert to Twikit format
    cookies = {c['name']: c['value'] for c in browser_cookies}
    
    # Create client and set cookies
    client = TwikitClient(language=config.get("language", "en-US"))
    client.set_cookies(cookies)
    
    print("✓ Cookies loaded!")
    
    print("\n" + "=" * 60)
    print("Testing Twitter Scraper")
    print("=" * 60)
    
    # Test 1: Get home timeline
    print("\n1. Fetching home timeline...")
    try:
        timeline = await client.get_latest_timeline(count=20)
        print(f"   ✓ Got {len(timeline)} tweets")
        for i, tweet in enumerate(timeline[:3], 1):
            print(f"   {i}. @{tweet.user.screen_name}: {tweet.text[:50]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 2: Search tweets
    print("\n2. Searching for 'Python'...")
    try:
        results = await client.search_tweet("Python", product="Top", count=10)
        print(f"   ✓ Got {len(results)} results")
        for i, tweet in enumerate(results[:3], 1):
            print(f"   {i}. @{tweet.user.screen_name}: {tweet.text[:50]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 3: Get trends
    print("\n3. Fetching trends...")
    try:
        trends = await client.get_trends(category="trending")
        print(f"   ✓ Got {len(trends)} trends")
        for i, trend in enumerate(trends[:5], 1):
            print(f"   {i}. {trend.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 4: Get user tweets
    print("\n4. Fetching tweets from @doomlord14686...")
    try:
        tweets = await client.get_user_tweets("doomlord14686", tweet_type="Tweets", count=5)
        print(f"   ✓ Got {len(tweets)} tweets")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_scraper())
