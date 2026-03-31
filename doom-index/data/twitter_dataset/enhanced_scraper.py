#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper with Media Download and Comments/Replies

This script scrapes Twitter/X data with:
- Comments/replies for each tweet
- Media download (photos/videos)
- Structured data with tags
- Proper categorization

Usage:
    python enhanced_scraper.py
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
TWEETS_PER_KEYWORD = 3000
MAX_REPLIES_PER_TWEET = 10000


def load_cookies() -> Dict[str, str]:
    """Load and convert cookies from browser export format."""
    with open(COOKIES_FILE, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert complex objects to serializable format
    def convert(obj):
        if hasattr(obj, '__dict__'):
            return convert(obj.__dict__)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)


def save_csv(data: List[Dict], filepath: Path) -> None:
    """Save data as CSV."""
    if not data:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    import csv
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


async def download_media(media_list: List[Any], tweet_id: str, media_dir: Path) -> List[Dict]:
    """Download photos/videos from a tweet."""
    downloaded = []
    
    for i, media in enumerate(media_list):
        try:
            if hasattr(media, 'type') and media.type == 'photo':
                # Download photo
                url = getattr(media, 'url', None) or getattr(media, 'media_url', None)
                if url:
                    ext = '.jpg'
                    if 'png' in url.lower():
                        ext = '.png'
                    elif 'gif' in url.lower():
                        ext = '.gif'
                    
                    filename = f"{tweet_id}_{i}{ext}"
                    filepath = media_dir / filename
                    
                    # Download using httpx
                    import httpx
                    response = httpx.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        downloaded.append({
                            'filename': filename,
                            'type': 'photo',
                            'url': url,
                            'local_path': str(filepath)
                        })
                        print(f"    📷 Downloaded: {filename}")
                        
        except Exception as e:
            print(f"    ⚠️ Media download error: {e}")
            continue
    
    return downloaded


async def get_tweet_replies(client: Client, tweet_id: str, max_replies: int = 10) -> List[Dict]:
    """Get replies to a tweet using search with conversation ID."""
    replies = []
    
    try:
        # Search for replies using conversation_id
        query = f"conversation_id:{tweet_id}"
        tweets = await client.search_tweet(query=query, product='Latest', count=max_replies)
        
        for t in tweets:
            # Filter to only actual replies (not the original tweet)
            if str(t.id) != str(tweet_id):
                replies.append({
                    'reply_id': t.id,
                    'user': getattr(t.user, 'username', None) if t.user else None,
                    'user_name': getattr(t.user, 'name', None) if t.user else None,
                    'text': t.text,
                    'created_at': str(t.created_at) if t.created_at else None,
                    'likes': t.favorite_count,
                    'retweets': t.retweet_count,
                })
                
    except Exception as e:
        print(f"    ⚠️ Error getting replies: {e}")
    
    return replies


def parse_tweet(tweet: TwikitTweet, keyword: str = None) -> Dict:
    """Parse a tweet into structured data."""
    # Extract hashtags
    hashtags = []
    if hasattr(tweet, 'hashtags') and tweet.hashtags:
        hashtags = tweet.hashtags
    
    # Extract media
    media_list = []
    media_urls = []
    if hasattr(tweet, 'media') and tweet.media:
        for m in tweet.media:
            media_list.append({
                'type': getattr(m, 'type', 'unknown'),
                'url': getattr(m, 'url', None),
            })
            if hasattr(m, 'url') and m.url:
                media_urls.append(m.url)
    
    # Extract user info
    user_data = {}
    if tweet.user:
        user_data = {
            'id': str(getattr(tweet.user, 'id', None)),
            'username': getattr(tweet.user, 'username', None),
            'name': getattr(tweet.user, 'name', None),
            'followers': getattr(tweet.user, 'followers_count', 0),
            'following': getattr(tweet.user, 'friends_count', 0),
            'verified': getattr(tweet.user, 'verified', False),
            'bio': getattr(tweet.user, 'description', None),
        }
    
    return {
        'tweet_id': str(tweet.id),
        'keyword': keyword,
        'text': tweet.text,
        'full_text': getattr(tweet, 'full_text', tweet.text),
        'created_at': str(tweet.created_at) if tweet.created_at else None,
        'user': user_data,
        'metrics': {
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count,
            'replies': tweet.reply_count,
            'quotes': getattr(tweet, 'quote_count', 0),
            'views': getattr(tweet, 'view_count', 0),
        },
        'hashtags': hashtags,
        'mentions': getattr(tweet, 'mentions', []),
        'urls': getattr(tweet, 'urls', []),
        'media': media_list,
        'media_urls': media_urls,
        'is_retweet': hasattr(tweet, 'retweeted_tweet') and tweet.retweeted_tweet,
        'is_reply': getattr(tweet, 'in_reply_to', None) is not None,
        'is_quote': getattr(tweet, 'is_quote_status', False),
        'language': getattr(tweet, 'lang', 'en'),
        'possibly_sensitive': getattr(tweet, 'possibly_sensitive', False),
    }


async def scrape_with_replies_and_media():
    """Main scraping function with replies and media."""
    print("=" * 60)
    print("ENHANCED TWITTER SCRAPER - WITH REPLIES & MEDIA")
    print("=" * 60)
    
    # Initialize client
    print("\n[1/4] Loading cookies and initializing client...")
    cookies = load_cookies()
    client = Client('en-US')
    client.set_cookies(cookies)
    print("✓ Client initialized!")
    
    # Create media directory
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Media directory: {MEDIA_DIR}")
    
    all_tweets = []
    all_replies = []
    media_count = 0
    
    # Scrape each keyword
    print(f"\n[2/4] Scraping {len(KEYWORDS)} keywords...")
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n  [{i+1}/{len(KEYWORDS)}] Searching: '{keyword}'")
        
        try:
            # Search tweets
            tweets = await client.search_tweet(query=keyword, product='Top', count=TWEETS_PER_KEYWORD)
            print(f"    Found {len(tweets)} tweets")
            
            for tweet in tweets:
                # Parse main tweet
                tweet_data = parse_tweet(tweet, keyword)
                all_tweets.append(tweet_data)
                
                # Download media
                if tweet.media:
                    print(f"    Downloading media for tweet {tweet.id}...")
                    downloaded = await download_media(tweet.media, str(tweet.id), MEDIA_DIR)
                    tweet_data['downloaded_media'] = downloaded
                    media_count += len(downloaded)
                
                # Get replies (limit to avoid rate limits)
                if tweet.reply_count and tweet.reply_count > 0:
                    print(f"    Getting replies for {tweet.id}...")
                    await asyncio.sleep(1)  # Rate limiting
                    replies = await get_tweet_replies(client, str(tweet.id), MAX_REPLIES_PER_TWEET)
                    
                    if replies:
                        print(f"    Found {len(replies)} replies")
                        tweet_data['replies_data'] = replies
                        
                        # Add to replies collection
                        for reply in replies:
                            reply['parent_tweet_id'] = str(tweet.id)
                            reply['parent_keyword'] = keyword
                            all_replies.append(reply)
                
                # Small delay between tweets
                await asyncio.sleep(0.5)
                
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            await asyncio.sleep(5)  # Wait on error
        
        # Delay between keywords
        await asyncio.sleep(2)
    
    # Also get timeline and trends
    print("\n[3/4] Fetching timeline and trends...")
    
    try:
        timeline = await client.get_latest_timeline(count=20)
        timeline_data = [parse_tweet(t, 'timeline') for t in timeline]
        all_tweets.extend(timeline_data)
        print(f"    Timeline: {len(timeline_data)} tweets")
    except Exception as e:
        print(f"    ⚠️ Timeline error: {e}")
    
    try:
        trends = await client.get_trends(category='trending')
        print(f"    Trends: {len(trends)} topics")
    except Exception as e:
        print(f"    ⚠️ Trends error: {e}")
        trends = []
    
    # Save all data
    print(f"\n[4/4] Saving data...")
    
    # Main tweets
    save_json(all_tweets, OUTPUT_DIR / 'enhanced_tweets.json')
    save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    # Replies
    save_json(all_replies, OUTPUT_DIR / 'replies.json')
    save_csv(all_replies, OUTPUT_DIR / 'replies.csv')
    
    # Summary
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'keywords_scraped': len(KEYWORDS),
        'total_tweets': len(all_tweets),
        'total_replies': len(all_replies),
        'media_downloaded': media_count,
        'media_directory': str(MEDIA_DIR),
    }
    save_json(summary, OUTPUT_DIR / 'scrape_summary.json')
    
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   - Tweets collected: {len(all_tweets)}")
    print(f"   - Replies collected: {len(all_replies)}")
    print(f"   - Media files: {media_count}")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"   - enhanced_tweets.json/csv")
    print(f"   - replies.json/csv")
    print(f"   - scrape_summary.json")
    print(f"   - media/ (downloaded photos/videos)")


if __name__ == '__main__':
    asyncio.run(scrape_with_replies_and_media())
