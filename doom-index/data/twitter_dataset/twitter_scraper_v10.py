#!/usr/bin/env python3
"""
Twitter Scraper V10 - Using twscrape

This version tries both cookie-based accounts AND guest token fallback.
"""

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from twscrape import API

# ============== CONFIGURATION ==============
OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing cookies
COOKIE_FILES = [
    'doom-index/src/data/scrapers/twitter_cookies.json',
    'doom-index/src/data/scrapers/twitter_cookies2.json',
]

# Keywords for controversy/cancel culture research  
KEYWORDS = [
    '#cancel', '#cancelled', '#cancellation', 'cancelled after', 
    'facing backlash', 'called out for', 'under fire for', 
    'controversy', 'boycott', 'petition', 'cancel culture',
    'backlash', 'outrage', 'offensive', 'scandal'
]

MAX_TWEETS_PER_KEYWORD = 5000


def load_cookies_str(path: str) -> str:
    """Load cookies from JSON file and convert to string format."""
    with open(path, 'r') as f:
        cookies_list = json.load(f)
    cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies_list])
    return cookie_str


def save_csv(tweets: list, filepath: Path):
    """Save tweets to CSV."""
    if not tweets:
        return
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(tweets[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tweets)


async def scrape_keyword(api: API, keyword: str, seen_ids: set) -> list:
    """Scrape tweets for a keyword."""
    all_tweets = []
    
    print(f"    Searching: '{keyword}'")
    
    count = 0
    try:
        async for tweet in api.search(keyword, limit=MAX_TWEETS_PER_KEYWORD):
            tid = str(tweet.id)
            
            if tid in seen_ids:
                continue
            
            seen_ids.add(tid)
            
            # Extract media URLs
            media_urls = []
            if hasattr(tweet, 'media') and tweet.media:
                for m in tweet.media:
                    if hasattr(m, 'url'):
                        media_urls.append(m.url)
            
            parsed = {
                'tweet_id': tid,
                'text': tweet.rawContent or "",
                'created_at': tweet.created_at.isoformat() if tweet.created_at else "",
                'user_id': str(tweet.user.id) if tweet.user else "",
                'username': tweet.user.username if tweet.user else "",
                'name': tweet.user.name if tweet.user else "",
                'followers': tweet.user.followers_count if tweet.user else 0,
                'verified': 1 if (tweet.user and tweet.user.verified) else 0,
                'likes': tweet.favorite_count or 0,
                'retweets': tweet.retweet_count or 0,
                'replies': tweet.reply_count or 0,
                'keyword': keyword,
                'media': ','.join(media_urls) if media_urls else "",
            }
            
            all_tweets.append(parsed)
            count += 1
            
            if count % 100 == 0:
                print(f"      +{count} tweets collected")
                
    except Exception as e:
        print(f"    ⚠️ Error: {str(e)[:80]}")
    
    print(f"    '{keyword}': +{count} tweets")
    return all_tweets


async def main():
    """Main scraping function."""
    print("=" * 60)
    print("TWITTER SCRAPER V10 - twscrape")
    print("=" * 60)
    
    # Initialize API
    api = API()
    
    # Add accounts from existing cookies
    print("\n[1/5] Setting up accounts...")
    
    for cf in COOKIE_FILES:
        try:
            cookies_str = load_cookies_str(cf)
            await api.pool.add_account(
                "doomlord14686", "Hesoyam1@", 
                "vaasha038@gmail.com", "Hesoyam1@",
                cookies=cookies_str
            )
            print(f"  ✓ Added {cf}")
        except Exception as e:
            pass  # Account may already exist
    
    # Try login
    print("\n[2/5] Logging in...")
    try:
        await api.pool.login_all()
        print("  ✓ Login done")
    except Exception as e:
        print(f"  ⚠️ Login issue (will try anyway): {str(e)[:50]}")
    
    # Load existing data
    print("\n[3/5] Loading existing data...")
    all_tweets = []
    seen_ids = set()
    
    tweets_file = OUTPUT_DIR / 'enhanced_tweets.csv'
    if tweets_file.exists():
        with open(tweets_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_tweets.append(row)
                seen_ids.add(row.get('tweet_id', ''))
    print(f"  Loaded {len(all_tweets)} tweets")
    
    # Try to scrape (even without active accounts, twscrape may get guest token)
    print(f"\n[4/5] Scraping {len(KEYWORDS)} keywords...")
    print("  Note: If no data, cookies are expired and need refresh")
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        try:
            tweets = await scrape_keyword(api, keyword, seen_ids)
            all_tweets.extend(tweets)
        except Exception as e:
            print(f"    ⚠️ Error: {str(e)[:60]}")
        
        if (i + 1) % 3 == 0:
            save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    # Save final
    print("\n[5/5] Saving...")
    save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'total_tweets': len(all_tweets),
    }
    with open(OUTPUT_DIR / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"📊 Total tweets: {len(all_tweets)}")
    print(f"📁 {OUTPUT_DIR}")
    
    if len(all_tweets) == len(existing_tweets) if 'existing_tweets' in dir() else 0:
        print("\n⚠️ No new tweets - cookies are likely expired!")
        print("To refresh: get fresh cookies from browser and update cookie files")


if __name__ == '__main__':
    asyncio.run(main())
