#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper V3 - Rate Limit Resistant

This version includes:
- Automatic rate limit detection and backoff
- Longer delays between requests
- Multiple cookie rotation (if multiple accounts provided)
- Progress saving to resume after rate limits
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add twikit path
sys.path.insert(0, '/Users/ajay/Library/PythonExtensions/lib/python3.9/site-packages')

from twikit import Client, Tweet as TwikitTweet

# Configuration
COOKIES_FILES = [
    'doom-index/src/data/scrapers/twitter_cookies.json',
]
OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
MEDIA_DIR = OUTPUT_DIR / 'media'

# Keywords to search
KEYWORDS = [
    '#cancel', '#cancelled', '#cancellation', 'cancelled after', 
    'facing backlash', 'called out for', 'under fire for', 
    'controversy', 'boycott', 'petition', 'cancel culture',
    'backlash', 'outrage', 'offensive'
]

# Settings
TWEETS_PER_KEYWORD = 2500
MAX_REPLIES_PER_TWEET = 5000
DELAY_BETWEEN_REQUESTS = 3  # seconds
DELAY_BETWEEN_KEYWORDS = 5  # seconds
RATE_LIMIT_BACKOFF = 30     # seconds to wait on rate limit


def load_cookies(cookies_file: str) -> Dict[str, str]:
    """Load and convert cookies from browser export format."""
    with open(cookies_file, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def create_client(cookies_file: str) -> Client:
    """Create and initialize a Twitter client with cookies."""
    cookies = load_cookies(cookies_file)
    client = Client('en-US')
    client.set_cookies(cookies)
    return client


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
    """Parse a tweet into basic structured data."""
    # Extract hashtags
    hashtags = []
    if hasattr(tweet, 'hashtags') and tweet.hashtags:
        hashtags = [safe_str(h) for h in tweet.hashtags]
    
    # Extract user info
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
    """Parse a reply tweet."""
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
        pass  # Silently handle
    
    return replies


async def download_media(media_list: List[Any], tweet_id: str, media_dir: Path) -> int:
    """Download photos from a tweet."""
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
            continue
    
    return downloaded


async def scrape_keyword(client: Client, keyword: str, media_dir: Path) -> tuple:
    """Scrape a single keyword - returns (tweets, replies, media_count)."""
    all_tweets = []
    all_replies = []
    media_count = 0
    
    try:
        tweets = await client.search_tweet(query=keyword, product='Top', count=TWEETS_PER_KEYWORD)
        print(f"    Found {len(tweets)} tweets")
        
        for tweet in tweets:
            # Parse tweet
            tweet_data = parse_tweet_basic(tweet, keyword)
            all_tweets.append(tweet_data)
            
            # Download media
            if hasattr(tweet, 'media') and tweet.media:
                count = await download_media(tweet.media, str(tweet.id), media_dir)
                media_count += count
            
            # Get replies
            if tweet.reply_count and tweet.reply_count > 0:
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
                replies = await get_tweet_replies(client, str(tweet.id), MAX_REPLIES_PER_TWEET)
                
                if replies:
                    for r in replies:
                        r['parent_keyword'] = keyword
                    all_replies.extend(replies)
            
            # Small delay between tweets
            await asyncio.sleep(0.5)
            
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg or 'Rate limit' in error_msg:
            print(f"    ⚠️ Rate limited! Waiting {RATE_LIMIT_BACKOFF}s...")
            await asyncio.sleep(RATE_LIMIT_BACKOFF)
            raise  # Re-raise to handle in main loop
        print(f"    ⚠️ Error: {e}")
    
    return all_tweets, all_replies, media_count


async def scrape_with_rate_limit_handling():
    """Main scraping function with rate limit handling."""
    print("=" * 60)
    print("ENHANCED TWITTER SCRAPER V3 - Rate Limit Resistant")
    print("=" * 60)
    
    # Initialize
    print("\n[1/4] Loading cookies...")
    clients = []
    for cf in COOKIES_FILES:
        try:
            client = create_client(cf)
            clients.append(client)
            print(f"  ✓ Loaded: {cf}")
        except Exception as e:
            print(f"  ✗ Failed: {cf} - {e}")
    
    if not clients:
        print("ERROR: No clients available!")
        return
    
    print(f"\n  Total clients: {len(clients)}")
    
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if any
    existing_tweets = []
    existing_replies = []
    
    tweets_file = OUTPUT_DIR / 'enhanced_tweets.csv'
    replies_file = OUTPUT_DIR / 'replies.csv'
    
    if tweets_file.exists():
        print("\n[2/4] Loading existing data...")
        import csv
        with open(tweets_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_tweets = list(reader)
        print(f"  Loaded {len(existing_tweets)} existing tweets")
    
    if replies_file.exists():
        with open(replies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_replies = list(reader)
        print(f"  Loaded {len(existing_replies)} existing replies")
    
    all_tweets = existing_tweets.copy()
    all_replies = existing_replies.copy()
    total_media = 0
    
    # Track which keywords are done
    done_keywords = set(t.get('keyword', '') for t in all_tweets if t.get('keyword'))
    
    # Scrape each keyword
    print(f"\n[3/4] Scraping {len(KEYWORDS)} keywords...")
    
    client_idx = 0
    for i, keyword in enumerate(KEYWORDS):
        if keyword in done_keywords:
            print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}' - SKIPPED (already done)")
            continue
            
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        # Rotate clients
        client = clients[client_idx % len(clients)]
        client_idx += 1
        
        # Retry logic
        max_retries = 3
        for retry in range(max_retries):
            try:
                await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
                tweets, replies, media_count = await scrape_keyword(client, keyword, MEDIA_DIR)
                
                all_tweets.extend(tweets)
                all_replies.extend(replies)
                total_media += media_count
                
                print(f"    +{len(tweets)} tweets, +{len(replies)} replies")
                break
                
            except Exception as e:
                if '429' in str(e) or 'Rate limit' in str(e):
                    wait_time = RATE_LIMIT_BACKOFF * (retry + 1)
                    print(f"    ⚠️ Rate limit! Waiting {wait_time}s... (attempt {retry+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    
                    # Try next client if available
                    if len(clients) > 1:
                        client = clients[client_idx % len(clients)]
                        client_idx += 1
                        print(f"    → Switching to next account...")
                else:
                    print(f"    ⚠️ Error: {e}")
                    break
        
        # Save progress periodically
        if (i + 1) % 5 == 0:
            print(f"\n  💾 Saving progress...")
            save_data(all_tweets, all_replies, OUTPUT_DIR)
    
    # Timeline
    print("\n[4/4] Fetching timeline...")
    try:
        client = clients[0]
        await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
        timeline = await client.get_latest_timeline(count=30)
        for t in timeline:
            all_tweets.append(parse_tweet_basic(t, 'timeline'))
        print(f"    Timeline: {len(timeline)} tweets")
    except Exception as e:
        print(f"    ⚠️ Timeline error: {e}")
    
    # Save final data
    print("\n💾 Saving final data...")
    save_data(all_tweets, all_replies, OUTPUT_DIR)
    
    # Summary
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'total_tweets': len(all_tweets),
        'total_replies': len(all_replies),
        'media_downloaded': total_media,
    }
    with open(OUTPUT_DIR / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f)
    
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   - Tweets: {len(all_tweets)}")
    print(f"   - Replies: {len(all_replies)}")
    print(f"   - Media files: {total_media}")
    print(f"\n📁 Files saved to: {OUTPUT_DIR}")


def save_data(tweets: List[Dict], replies: List[Dict], output_dir: Path):
    """Save tweets and replies to CSV files."""
    import csv
    
    # Tweets
    if tweets:
        with open(output_dir / 'enhanced_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=tweets[0].keys())
            writer.writeheader()
            writer.writerows(tweets)
    
    # Replies
    if replies:
        with open(output_dir / 'replies.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=replies[0].keys())
            writer.writeheader()
            writer.writerows(replies)


if __name__ == '__main__':
    asyncio.run(scrape_with_rate_limit_handling())
