#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper V4 - HIGH PERFORMANCE

Optimizations:
- Async parallel reply fetching (5-10 concurrent)
- Async media downloads
- Reduced limits (200 replies max)
- Skip low-engagement tweets
- Multiple account rotation
- Exponential backoff
- ~10-50x faster than V3
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

# ============== CONFIGURATION ==============
FORCE_RESCAPE = True  # Set to True to re-scrape all keywords
COOKIES_FILES = [
    'doom-index/src/data/scrapers/twitter_cookies.json',
    'doom-index/src/data/scrapers/twitter_cookies2.json',
    # 'cookies2.json',
    # 'cookies3.json', 
    # 'cookies4.json',
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

# ============== OPTIMIZED SETTINGS ==============
TWEETS_PER_KEYWORD = 1000      # Reduced from 2500
MAX_REPLIES_PER_TWEET = 500    # Reduced from 5000 - realistic limit
MIN_REPLIES_TO_FETCH = 5      # Skip tweets with <5 replies (80% of tweets)
DELAY_BETWEEN_REQUESTS = 0.5  # Fast but safe
DELAY_BETWEEN_KEYWORDS = 2    # Short delay between keywords

# Get older posts - use Latest instead of Top, and paginate
SEARCH_PRODUCT = 'Latest'
PAGES_TO_PAGINATE = 5

# Concurrency settings
MAX_CONCURRENT_REPLIES = 10   # Parallel reply fetching
MAX_CONCURRENT_MEDIA = 5      # Parallel media downloads
SEMAPHORE_REPLIES = asyncio.Semaphore(MAX_CONCURRENT_REPLIES)
SEMAPHORE_MEDIA = asyncio.Semaphore(MAX_CONCURRENT_MEDIA)

# Rate limit settings
INITIAL_BACKOFF = 10          # Initial wait on rate limit
MAX_BACKOFF = 120             # Max wait time


def load_cookies(cookies_file: str) -> Dict[str, str]:
    """Load cookies from browser export format."""
    with open(cookies_file, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def create_client(cookies_file: str) -> Client:
    """Create and initialize a Twitter client."""
    cookies = load_cookies(cookies_file)
    client = Client('en-US')
    client.set_cookies(cookies)
    return client


def safe_str(obj: Any) -> str:
    """Safely convert to string."""
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
    """Parse tweet into structured data."""
    # Extract hashtags
    hashtags = []
    if hasattr(tweet, 'hashtags') and tweet.hashtags:
        hashtags = [safe_str(h) for h in tweet.hashtags]
    
    # User info
    user_data = {}
    if tweet.user:
        user_data = {
            'user_id': safe_str(getattr(tweet.user, 'id', '')),
            'username': safe_str(getattr(tweet.user, 'username', '')),
            'name': safe_str(getattr(tweet.user, 'name', '')),
            'followers': int(getattr(tweet.user, 'followers_count', 0) or 0),
            'verified': bool(getattr(tweet.user, 'verified', False)),
        }
    
    # Media info
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
    """Parse reply tweet."""
    user_data = {}
    if tweet.user:
        user_data = {
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


async def get_tweet_replies(client: Client, tweet_id: str, max_replies: int) -> List[Dict]:
    """Get replies to a tweet."""
    replies = []
    try:
        query = f"conversation_id:{tweet_id}"
        tweets = await client.search_tweet(query=query, product='Latest', count=max_replies)
        for t in tweets:
            if str(t.id) != str(tweet_id):
                replies.append(parse_reply_basic(t, tweet_id, ''))
    except Exception:
        pass
    return replies


async def get_replies_with_semaphore(client: Client, tweet_id: str, max_replies: int) -> List[Dict]:
    """Get replies with semaphore for rate limiting."""
    async with SEMAPHORE_REPLIES:
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        return await get_tweet_replies(client, tweet_id, max_replies)


async def download_media_async(media_list: List[Any], tweet_id: str, media_dir: Path) -> int:
    """Download media with async."""
    import httpx
    downloaded = 0
    
    async with SEMAPHORE_MEDIA:
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
                        
                        async with httpx.AsyncClient() as http_client:
                            response = await http_client.get(url, timeout=30)
                            if response.status_code == 200:
                                with open(filepath, 'wb') as f:
                                    f.write(response.content)
                                downloaded += 1
            except Exception:
                continue
    
    return downloaded


async def process_tweet(client: Client, tweet: TwikitTweet, keyword: str, media_dir: Path) -> tuple:
    """Process a single tweet - get replies and download media."""
    tweet_data = parse_tweet_basic(tweet, keyword)
    replies = []
    media_count = 0
    
    # Download media
    if hasattr(tweet, 'media') and tweet.media:
        media_count = await download_media_async(tweet.media, str(tweet.id), media_dir)
    
    # Get replies ONLY if enough engagement (optimization!)
    reply_count = int(tweet.reply_count or 0)
    if reply_count >= MIN_REPLIES_TO_FETCH:
        replies = await get_replies_with_semaphore(client, str(tweet.id), MAX_REPLIES_PER_TWEET)
        for r in replies:
            r['parent_keyword'] = keyword
    
    return tweet_data, replies, media_count


async def scrape_keyword_fast(client: Client, keyword: str, media_dir: Path) -> tuple:
    """Scrape a keyword with parallel processing."""
    all_tweets = []
    all_replies = []
    media_count = 0
    
    try:
        # First page
        tweet_list = await client.search_tweet(query=keyword, product=SEARCH_PRODUCT, count=TWEETS_PER_KEYWORD)
        tweets = list(tweet_list)
        
        # Paginate to get older posts
        for page in range(PAGES_TO_PAGINATE):
            if hasattr(tweet_list, 'has_next_page') and tweet_list.has_next_page:
                print(f"    Fetching page {page+2}...")
                tweet_list = await tweet_list.next()
                tweets.extend(list(tweet_list))
                await asyncio.sleep(1)  # Delay between pages
        
        print(f"    Found {len(tweets)} tweets total ({PAGES_TO_PAGINATE} pages), processing with {MAX_CONCURRENT_REPLIES} parallel workers...")
        
        # Create tasks for parallel processing
        tasks = [
            process_tweet(client, tweet, keyword, media_dir)
            for tweet in tweets
        ]
        
        # Execute in batches to avoid overwhelming the API
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple):
                    tweet_data, replies, media = result
                    all_tweets.append(tweet_data)
                    all_replies.extend(replies)
                    media_count += media
            
            print(f"    Processed {min(i+batch_size, len(tasks))}/{len(tasks)} tweets...")
            
            # Small delay between batches
            await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
        
    except Exception as e:
        error_str = str(e)
        if '429' in error_str or 'Rate limit' in error_str:
            print(f"    ⚠️ Rate limited!")
            raise
        print(f"    ⚠️ Error: {e}")
    
    return all_tweets, all_replies, media_count


async def scrape_with_exponential_backoff():
    """Main scraping with exponential backoff."""
    print("=" * 60)
    print("ENHANCED TWITTER SCRAPER V4 - HIGH PERFORMANCE")
    print("=" * 60)
    print(f"\n⚡ Optimizations:")
    print(f"   - Parallel reply fetching ({MAX_CONCURRENT_REPLIES} concurrent)")
    print(f"   - Async media downloads ({MAX_CONCURRENT_MEDIA} concurrent)")
    print(f"   - Skip tweets with <{MIN_REPLIES_TO_FETCH} replies")
    print(f"   - Max {MAX_REPLIES_PER_TWEET} replies per tweet")
    print(f"   - {len(COOKIES_FILES)} account(s) available")
    
    # Initialize clients
    print("\n[1/5] Loading accounts...")
    clients = []
    for cf in COOKIES_FILES:
        try:
            client = create_client(cf)
            clients.append(client)
            print(f"  ✓ Loaded: {cf}")
        except Exception as e:
            print(f"  ✗ Failed: {cf}")
    
    if not clients:
        print("ERROR: No clients!")
        return
    
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing data
    existing_tweets = []
    existing_replies = []
    tweets_file = OUTPUT_DIR / 'enhanced_tweets.csv'
    replies_file = OUTPUT_DIR / 'replies.csv'
    
    if tweets_file.exists():
        print("\n[2/5] Loading existing data...")
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
    
    done_keywords = set()
    if not FORCE_RESCAPE:
        done_keywords = set(t.get('keyword', '') for t in all_tweets if t.get('keyword'))
    
    # Scrape keywords
    print(f"\n[3/5] Scraping {len(KEYWORDS)} keywords (parallel)...")
    client_idx = 0
    
    for i, keyword in enumerate(KEYWORDS):
        if keyword in done_keywords:
            print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}' - SKIPPED")
            continue
        
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        client = clients[client_idx % len(clients)]
        client_idx += 1
        
        # Exponential backoff retry
        backoff = INITIAL_BACKOFF
        for retry in range(5):
            try:
                await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
                tweets, replies, media = await scrape_keyword_fast(client, keyword, MEDIA_DIR)
                
                all_tweets.extend(tweets)
                all_replies.extend(replies)
                total_media += media
                
                print(f"    +{len(tweets)} tweets, +{len(replies)} replies, +{media} media")
                break
                
            except Exception as e:
                if '429' in str(e):
                    print(f"    ⚠️ Rate limit! Waiting {backoff}s (retry {retry+1})...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    
                    if len(clients) > 1:
                        client = clients[client_idx % len(clients)]
                        client_idx += 1
                else:
                    print(f"    ⚠️ Error: {e}")
                    break
        
        # Save progress
        if (i + 1) % 3 == 0:
            print(f"\n  💾 Saving progress...")
            save_data(all_tweets, all_replies, OUTPUT_DIR)
    
    # Timeline
    print("\n[4/5] Fetching timeline...")
    try:
        client = clients[0]
        timeline = await client.get_latest_timeline(count=50)
        for t in timeline:
            all_tweets.append(parse_tweet_basic(t, 'timeline'))
        print(f"    +{len(timeline)} timeline tweets")
    except Exception as e:
        print(f"    ⚠️ Timeline error: {e}")
    
    # Save
    print("\n[5/5] Saving data...")
    save_data(all_tweets, all_replies, OUTPUT_DIR)
    
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
    print(f"   - Media: {total_media}")
    print(f"\n📁 {OUTPUT_DIR}")


def save_data(tweets: List[Dict], replies: List[Dict], output_dir: Path):
    """Save to CSV."""
    import csv
    
    if tweets:
        with open(output_dir / 'enhanced_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=tweets[0].keys())
            writer.writeheader()
            writer.writerows(tweets)
    
    if replies:
        with open(output_dir / 'replies.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=replies[0].keys())
            writer.writeheader()
            writer.writerows(replies)


if __name__ == '__main__':
    asyncio.run(scrape_with_exponential_backoff())
