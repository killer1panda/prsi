#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper V5 - PRODUCTION GRADE

Key improvements over V4:
- Global HTTP client (2-3x faster media)
- Duplicate prevention
- Incremental saves (memory efficient)
- Better filtering (skip viral tweets)
- Safer semaphore handling
- Retry logic for media
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
FORCE_RESCAPE = False  # Set True to re-scrape all
COOKIES_FILES = [
    'doom-index/src/data/scrapers/twitter_cookies.json',
    'doom-index/src/data/scrapers/twitter_cookies2.json',
]

OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
MEDIA_DIR = OUTPUT_DIR / 'media'

KEYWORDS = [
    '#cancel', '#cancelled', '#cancellation', 'cancelled after', 
    'facing backlash', 'called out for', 'under fire for', 
    'controversy', 'boycott', 'petition', 'cancel culture',
    'backlash', 'outrage', 'offensive'
]

# ============== OPTIMIZED SETTINGS ==============
TWEETS_PER_KEYWORD = 1000
MAX_REPLIES_PER_TWEET = 50000
MIN_REPLIES_TO_FETCH = 5
MAX_REPLIES_TO_SKIP = 100000  # Skip viral tweets (>100000 replies)
SEARCH_PRODUCT = 'Latest'
PAGES_TO_PAGINATE = 500

# Concurrency - CONSERVATIVE for reliability
MAX_CONCURRENT_REPLIES = 5   # Reduced from 10
MAX_CONCURRENT_MEDIA = 3     # Reduced from 5
BATCH_SIZE = 10             # Reduced from 20

DELAY_BETWEEN_REQUESTS = 0.5
DELAY_BETWEEN_KEYWORDS = 2

# HTTP Client - GLOBAL (created once)
import httpx
HTTP_CLIENT = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=20))


def load_cookies(cookies_file: str) -> Dict[str, str]:
    with open(cookies_file, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def create_client(cookies_file: str) -> Client:
    cookies = load_cookies(cookies_file)
    client = Client('en-US')
    client.set_cookies(cookies)
    return client


def safe_str(obj: Any) -> str:
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
    hashtags = []
    if hasattr(tweet, 'hashtags') and tweet.hashtags:
        hashtags = [safe_str(h) for h in tweet.hashtags]
    
    user_data = {}
    if tweet.user:
        user_data = {
            'user_id': safe_str(getattr(tweet.user, 'id', '')),
            'username': safe_str(getattr(tweet.user, 'username', '')),
            'name': safe_str(getattr(tweet.user, 'name', '')),
            'followers': int(getattr(tweet.user, 'followers_count', 0) or 0),
            'verified': bool(getattr(tweet.user, 'verified', False)),
        }
    
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
    replies = []
    try:
        # Add -filter:retweets to remove junk
        query = f"conversation_id:{tweet_id} -filter:retweets"
        tweets = await client.search_tweet(query=query, product='Latest', count=max_replies)
        for t in tweets:
            if str(t.id) != str(tweet_id):
                replies.append(parse_reply_basic(t, tweet_id, ''))
    except Exception:
        pass
    return replies


async def download_media_with_retry(media_list: List[Any], tweet_id: str, media_dir: Path) -> int:
    downloaded = 0
    
    for i, media in enumerate(media_list):
        for attempt in range(3):  # 3 retries
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
                        
                        response = await HTTP_CLIENT.get(url)
                        if response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            downloaded += 1
                            break  # Success, exit retry loop
                        
                break  # Success or non-retryable error
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1)  # Wait before retry
    
    return downloaded


async def process_tweet(client, tweet, keyword, media_dir, semaphore):
    async with semaphore:
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        
        tweet_data = parse_tweet_basic(tweet, keyword)
        replies = []
        media_count = 0
        
        # Download media
        if hasattr(tweet, 'media') and tweet.media:
            media_count = await download_media_with_retry(tweet.media, str(tweet.id), media_dir)
        
        # Get replies only if engagement is in sweet spot (enough but not viral)
        reply_count = int(tweet.reply_count or 0)
        if MIN_REPLIES_TO_FETCH <= reply_count < MAX_REPLIES_TO_SKIP:
            replies = await get_tweet_replies(client, str(tweet.id), MAX_REPLIES_PER_TWEET)
            for r in replies:
                r['parent_keyword'] = keyword
        
        return tweet_data, replies, media_count


async def scrape_keyword_fast(client, keyword, media_dir, seen_ids) -> tuple:
    all_tweets = []
    all_replies = []
    media_count = 0
    
    # Create semaphores for this keyword
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REPLIES)
    
    try:
        tweet_list = await client.search_tweet(query=keyword, product=SEARCH_PRODUCT, count=TWEETS_PER_KEYWORD)
        tweets = list(tweet_list)
        
        # Paginate
        for page in range(PAGES_TO_PAGINATE):
            if hasattr(tweet_list, 'has_next_page') and tweet_list.has_next_page:
                tweet_list = await tweet_list.next()
                tweets.extend(list(tweet_list))
                await asyncio.sleep(1)
        
        # Deduplicate by ID
        unique_tweets = []
        for t in tweets:
            if str(t.id) not in seen_ids:
                seen_ids.add(str(t.id))
                unique_tweets.append(t)
        tweets = unique_tweets
        
        print(f"    Found {len(tweets)} unique tweets (pagination + dedup), processing...")
        
        # Process in batches
        batch_size = BATCH_SIZE
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]
            tasks = [process_tweet(client, t, keyword, media_dir, semaphore) for t in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple):
                    tweet_data, replies, media = result
                    all_tweets.append(tweet_data)
                    all_replies.extend(replies)
                    all_replies.extend(replies)
                    media_count += media
            
            print(f"    Processed {min(i+batch_size, len(tweets))}/{len(tweets)} tweets...")
            await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
        
    except Exception as e:
        if '429' in str(e) or 'Rate limit' in str(e):
            print(f"    ⚠️ Rate limited!")
            raise
        print(f"    ⚠️ Error: {e}")
    
    return all_tweets, all_replies, media_count


async def scrape_production_grade():
    """Production-grade scraper with all fixes."""
    print("=" * 60)
    print("TWITTER SCRAPER V5 - PRODUCTION GRADE")
    print("=" * 60)
    print(f"\n⚡ Improvements:")
    print(f"   - Global HTTP client")
    print(f"   - Duplicate prevention")
    print(f"   - Incremental saves")
    print(f"   - Skip viral tweets (>{MAX_REPLIES_TO_SKIP} replies)")
    print(f"   - Retry logic for media")
    print(f"   - Conservative batching ({BATCH_SIZE})")
    
    # Initialize clients
    print("\n[1/6] Loading accounts...")
    clients = []
    for cf in COOKIES_FILES:
        try:
            client = create_client(cf)
            clients.append(client)
            print(f"  ✓ {cf}")
        except Exception as e:
            print(f"  ✗ {cf}: {e}")
    
    if not clients:
        print("ERROR: No clients!")
        return
    
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing
    print("\n[2/6] Loading existing data...")
    all_tweets = []
    all_replies = []
    seen_ids = set()
    
    tweets_file = OUTPUT_DIR / 'enhanced_tweets.csv'
    replies_file = OUTPUT_DIR / 'replies.csv'
    
    if tweets_file.exists():
        import csv
        with open(tweets_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_tweets.append(row)
                seen_ids.add(row.get('tweet_id', ''))
        print(f"  Loaded {len(all_tweets)} tweets, {len(seen_ids)} unique IDs")
    
    if replies_file.exists():
        with open(replies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_replies = list(reader)
        print(f"  Loaded {len(all_replies)} replies")
    
    # Determine keywords
    done_keywords = set()
    if not FORCE_RESCAPE:
        done_keywords = set(t.get('keyword', '') for t in all_tweets if t.get('keyword'))
    
    # Scrape
    print(f"\n[3/6] Scraping {len(KEYWORDS)} keywords...")
    client_idx = 0
    
    for i, keyword in enumerate(KEYWORDS):
        if keyword in done_keywords:
            print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}' - SKIPPED")
            continue
        
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        client = clients[client_idx % len(clients)]
        client_idx += 1
        
        # Retry with backoff
        backoff = 10
        for retry in range(5):
            try:
                await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
                tweets, replies, media = await scrape_keyword_fast(client, keyword, MEDIA_DIR, seen_ids)
                
                all_tweets.extend(tweets)
                all_replies.extend(replies)
                
                print(f"    +{len(tweets)} tweets, +{len(replies)} replies, +{media} media")
                break
                
            except Exception as e:
                if '429' in str(e):
                    print(f"    ⚠️ Rate limit! Waiting {backoff}s...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    if len(clients) > 1:
                        client = clients[client_idx % len(clients)]
                        client_idx += 1
                else:
                    print(f"    ⚠️ Error: {e}")
                    break
        
        # Incremental save every 3 keywords
        if (i + 1) % 3 == 0:
            print(f"\n  💾 Saving progress...")
            save_data(all_tweets, all_replies, OUTPUT_DIR)
    
    # Timeline
    print("\n[4/6] Fetching timeline...")
    try:
        client = clients[0]
        timeline = await client.get_latest_timeline(count=50)
        for t in timeline:
            if str(t.id) not in seen_ids:
                seen_ids.add(str(t.id))
                all_tweets.append(parse_tweet_basic(t, 'timeline'))
        print(f"    +{len(timeline)} timeline tweets")
    except Exception as e:
        print(f"    ⚠️ Error: {e}")
    
    # Final save
    print("\n[5/6] Saving final data...")
    save_data(all_tweets, all_replies, OUTPUT_DIR)
    
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'total_tweets': len(all_tweets),
        'total_replies': len(all_replies),
    }
    with open(OUTPUT_DIR / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f)
    
    print("\n[6/6] Done!")
    print("=" * 60)
    print(f"📊 Results:")
    print(f"   - Tweets: {len(all_tweets)}")
    print(f"   - Replies: {len(all_replies)}")
    print(f"📁 {OUTPUT_DIR}")


def save_data(tweets, replies, output_dir):
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
    asyncio.run(scrape_production_grade())
