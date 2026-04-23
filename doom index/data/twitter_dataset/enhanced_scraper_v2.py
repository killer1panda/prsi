#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper v2 - Simplified JSON/CSV export

This version focuses on getting data with replies and media,
but simplifies the export to avoid complex object serialization.
"""

import asyncio
import json
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add twikit path
sys.path.insert(0, '/Users/ajay/Library/PythonExtensions/lib/python3.9/site-packages')

from twikit import Client, Tweet as TwikitTweet

# Configuration
COOKIES_FILE = 'doom-index/src/data/scrapers/twitter_cookies.json'
OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
MEDIA_DIR = OUTPUT_DIR / 'media'
KEYWORDS = [
    '#cancel', '#cancelled', '#cancellation', 'cancelled after', 
    'facing backlash', 'called out for', 'under fire for', 
    'controversy', 'boycott', 'petition', 'cancel culture',
    'backlash', 'outrage', 'offensive'
]
TWEETS_PER_KEYWORD = 2000
MAX_REPLIES_PER_TWEET = 5000


def load_cookies() -> Dict[str, str]:
    """Load and convert cookies from browser export format."""
    with open(COOKIES_FILE, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def safe_str(obj: Any) -> str:
    """Safely convert any object to string."""
    if obj is None:
        return ''
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if hasattr(obj, '__dict__'):
        return str(obj)
    return str(obj)


def parse_tweet_basic(tweet: TwikitTweet, keyword: str = None) -> Dict:
    """Parse a tweet into basic structured data (no complex objects)."""
    # Extract hashtags
    hashtags = []
    if hasattr(tweet, 'hashtags') and tweet.hashtags:
        hashtags = [safe_str(h) for h in tweet.hashtags]
    
    # Extract user info safely
    user_data = {}
    if tweet.user:
        user_data = {
            'user_id': safe_str(getattr(tweet.user, 'id', '')),
            'username': safe_str(getattr(tweet.user, 'username', '')),
            'name': safe_str(getattr(tweet.user, 'name', '')),
            'followers': int(getattr(tweet.user, 'followers_count', 0) or 0),
            'verified': bool(getattr(tweet.user, 'verified', False)),
        }
    
    # Extract media info
    media_list = []
    if hasattr(tweet, 'media') and tweet.media:
        for m in tweet.media:
            media_list.append({
                'type': safe_str(getattr(m, 'type', 'unknown')),
                'url': safe_str(getattr(m, 'url', '')),
            })
    
    return {
        'tweet_id': safe_str(tweet.id),
        'keyword': keyword,
        'text': safe_str(tweet.text),
        'created_at': safe_str(tweet.created_at),
        'user_id': user_data.get('user_id', ''),
        'username': user_data.get('username', ''),
        'name': user_data.get('name', ''),
        'followers': user_data.get('followers', 0),
        'verified': user_data.get('verified', False),
        'likes': int(tweet.favorite_count or 0),
        'retweets': int(tweet.retweet_count or 0),
        'replies': int(tweet.reply_count or 0),
        'hashtags': ' '.join(hashtags),
        'media_count': len(media_list),
        'media_info': json.dumps(media_list),
        'is_reply': bool(getattr(tweet, 'in_reply_to', None)),
        'is_retweet': bool(getattr(tweet, 'retweeted_tweet', None)),
    }


def parse_reply_basic(tweet: TwikitTweet, parent_id: str, keyword: str) -> Dict:
    """Parse a reply tweet into basic data."""
    user_data = {}
    if tweet.user:
        user_data = {
            'user_id': safe_str(getattr(tweet.user, 'id', '')),
            'username': safe_str(getattr(tweet.user, 'username', '')),
            'name': safe_str(getattr(tweet.user, 'name', '')),
        }
    
    return {
        'reply_id': safe_str(tweet.id),
        'parent_tweet_id': parent_id,
        'parent_keyword': keyword,
        'text': safe_str(tweet.text),
        'created_at': safe_str(tweet.created_at),
        'username': user_data.get('username', ''),
        'name': user_data.get('name', ''),
        'likes': int(tweet.favorite_count or 0),
        'retweets': int(tweet.retweet_count or 0),
    }


async def download_media(media_list: List[Any], tweet_id: str, media_dir: Path) -> int:
    """Download photos from a tweet. Returns count of downloaded files."""
    import httpx
    downloaded = 0
    
    for i, media in enumerate(media_list):
        try:
            if hasattr(media, 'type') and media.type == 'photo':
                url = getattr(media, 'url', None)
                if url:
                    ext = '.jpg'
                    if 'png' in url.lower():
                        ext = '.png'
                    elif 'gif' in url.lower():
                        ext = '.gif'
                    
                    filename = f"{tweet_id}_{i}{ext}"
                    filepath = media_dir / filename
                    
                    response = httpx.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        downloaded += 1
                        
        except Exception as e:
            print(f"    ⚠️ Media error: {e}")
            continue
    
    return downloaded


async def get_tweet_replies(client: Client, tweet_id: str, max_replies: int = 5) -> List[Dict]:
    """Get replies to a tweet."""
    replies = []
    
    try:
        query = f"conversation_id:{tweet_id}"
        tweets = await client.search_tweet(query=query, product='Latest', count=max_replies)
        
        for t in tweets:
            if str(t.id) != str(tweet_id):
                replies.append(parse_reply_basic(t, tweet_id, ''))
                
    except Exception as e:
        pass  # Silently handle errors
    
    return replies


async def scrape_with_replies_and_media():
    """Main scraping function."""
    print("=" * 60)
    print("ENHANCED TWITTER SCRAPER V2")
    print("=" * 60)
    
    # Initialize
    print("\n[1/4] Loading cookies...")
    cookies = load_cookies()
    client = Client('en-US')
    client.set_cookies(cookies)
    print("✓ Client ready")
    
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_tweets = []
    all_replies = []
    media_count = 0
    
    # Scrape keywords
    print(f"\n[2/4] Scraping {len(KEYWORDS)} keywords...")
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        try:
            tweets = await client.search_tweet(query=keyword, product='Top', count=TWEETS_PER_KEYWORD)
            print(f"    Found {len(tweets)} tweets")
            
            for tweet in tweets:
                # Parse tweet
                tweet_data = parse_tweet_basic(tweet, keyword)
                tweet_data['parent_keyword'] = keyword
                all_tweets.append(tweet_data)
                
                # Download media
                if hasattr(tweet, 'media') and tweet.media:
                    count = await download_media(tweet.media, str(tweet.id), MEDIA_DIR)
                    media_count += count
                    if count > 0:
                        print(f"    📷 Downloaded {count} media")
                
                # Get replies (limited)
                if tweet.reply_count and tweet.reply_count > 0:
                    await asyncio.sleep(0.5)
                    replies = await get_tweet_replies(client, str(tweet.id), MAX_REPLIES_PER_TWEET)
                    
                    if replies:
                        print(f"    💬 {len(replies)} replies")
                        for r in replies:
                            r['parent_keyword'] = keyword
                        all_replies.extend(replies)
                
                await asyncio.sleep(0.3)
                
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            await asyncio.sleep(5)
        
        await asyncio.sleep(2)
    
    # Timeline
    print("\n[3/4] Fetching timeline...")
    try:
        timeline = await client.get_latest_timeline(count=20)
        for t in timeline:
            all_tweets.append(parse_tweet_basic(t, 'timeline'))
        print(f"    Timeline: {len(timeline)} tweets")
    except Exception as e:
        print(f"    ⚠️ Timeline error: {e}")
    
    # Save data
    print(f"\n[4/4] Saving data...")
    
    # Tweets CSV
    if all_tweets:
        import csv
        with open(OUTPUT_DIR / 'enhanced_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_tweets[0].keys())
            writer.writeheader()
            writer.writerows(all_tweets)
    
    # Replies CSV
    if all_replies:
        import csv
        with open(OUTPUT_DIR / 'replies.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_replies[0].keys())
            writer.writeheader()
            writer.writerows(all_replies)
    
    # Summary
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'total_tweets': len(all_tweets),
        'total_replies': len(all_replies),
        'media_downloaded': media_count,
    }
    with open(OUTPUT_DIR / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f)
    
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   - Tweets: {len(all_tweets)}")
    print(f"   - Replies: {len(all_replies)}")
    print(f"   - Media files: {media_count}")
    print(f"\n📁 Files saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    asyncio.run(scrape_with_replies_and_media())
