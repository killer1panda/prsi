#!/usr/bin/env python3
"""Quick test of Twitter scraper search functionality"""
import asyncio
import sys
import os

# Add path for twikit
sys.path.insert(0, '/Users/ajay/Library/PythonExtensions/lib/python3.9/site-packages')

import twikit
from twikit import Client
import json

COOKIES_FILE = 'doom-index/src/data/scrapers/twitter_cookies.json'

async def main():
    # Load cookies
    with open(COOKIES_FILE, 'r') as f:
        cookies_list = json.load(f)
    
    # Convert from browser export format to dict format for twikit
    cookies_dict = {c['name']: c['value'] for c in cookies_list}
    
    # Create client and load cookies
    client = Client('en-US')
    client.set_cookies(cookies_dict)
    
    print("Testing search functionality...")
    
    # Try search
    try:
        tweets = await client.search_tweet('cancel', product='Top', count=30)
        print(f"Found {len(tweets)} tweets for 'cancel'")
        
        for tweet in tweets[:5]:
            print(f"  - {tweet.text[:100]}...")
            print(f"    Likes: {tweet.favorite_count}, RTs: {tweet.retweet_count}")
    except Exception as e:
        print(f"Search error: {e}")
    
    await client.close()

if __name__ == '__main__':
    asyncio.run(main())
