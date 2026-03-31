"""
Test script for the updated TwitterScraper.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from src.data.scrapers.twitter_scraper import TwitterScraper, TwikitConfig

# Create scraper with cookies file
config = TwikitConfig(
    cookies_file="src/data/scrapers/twitter_cookies.json",
    language="en-US"
)

print("Initializing TwitterScraper with cookies...")
scraper = TwitterScraper(twikit_config=config)

print("\n" + "=" * 60)
print("Testing Twitter Scraper")
print("=" * 60)

# Test 1: Get home timeline
print("\n1. Fetching home timeline...")
try:
    timeline = scraper.get_home_timeline(count=20)
    print(f"   ✓ Got {len(timeline)} tweets")
    for i, tweet in enumerate(timeline[:3], 1):
        print(f"   {i}. @{tweet['user']}: {tweet['text'][:50]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Search tweets
print("\n2. Searching for 'Python'...")
try:
    results = scraper.search_tweets("Python", product="Top", count=10)
    print(f"   ✓ Got {len(results)} results")
    for i, tweet in enumerate(results[:3], 1):
        print(f"   {i}. @{tweet['user']}: {tweet['text'][:50]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Get trends
print("\n3. Fetching trends...")
try:
    trends = scraper.get_trends(location="trending")
    print(f"   ✓ Got {len(trends)} trends")
    for i, trend in enumerate(trends[:5], 1):
        print(f"   {i}. {trend['name']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Get user tweets
print("\n4. Fetching tweets from @doomlord14686...")
try:
    tweets = scraper.get_user_timeline("doomlord14686", max_tweets=5)
    print(f"   ✓ Got {len(tweets)} tweets")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
