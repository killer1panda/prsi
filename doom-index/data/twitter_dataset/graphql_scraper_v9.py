#!/usr/bin/env python3
"""
Twitter GraphQL Scraper - Using Guest Token
This approach first gets a guest token, then uses it for GraphQL.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import httpx

# ============== CONFIGURATION ==============
OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing cookies
COOKIE_FILES = [
    'doom-index/src/data/scrapers/twitter_cookies.json',
    'doom-index/src/data/scrapers/twitter_cookies2.json',
]

KEYWORDS = [
    '#cancel', '#cancelled', '#cancellation', 'cancelled after', 
    'facing backlash', 'called out for', 'under fire for', 
    'controversy', 'boycott', 'petition', 'cancel culture',
    'backlash', 'outrage', 'offensive'
]

MAX_TWEETS = 10000
TWEETS_PER_PAGE = 50
MAX_PAGES = 20


def load_cookies(path: str) -> dict:
    """Load cookies from JSON file."""
    with open(path, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


async def get_guest_token(session: httpx.AsyncClient) -> str:
    """Get a guest token from Twitter."""
    
    # First, visit the homepage to get initial cookies
    try:
        await session.get('https://x.com/', timeout=10)
        await asyncio.sleep(1)
    except:
        pass
    
    # Get guest token
    url = "https://api.x.com/1.1/guest/activate.json"
    
    try:
        response = await session.post(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('guest_token', '')
    except Exception as e:
        print(f"    Guest token error: {e}")
    
    return None


async def search_tweets(session: httpx.AsyncClient, keyword: str, cursor: str = None) -> dict:
    """Search using Twitter's GraphQL API with guest token."""
    
    # GraphQL search endpoint with query ID
    url = "https://api.x.com/graphql/1wjM72S3U5JIsD7Y8G6Qw/SearchTimeline"
    
    variables = {
        'rawQuery': keyword,
        'count': TWEETS_PER_PAGE,
        'cursor': cursor,
        'querySource': 'typed_query',
        'product': 'Top'
    }
    
    params = {
        'variables': json.dumps(variables),
    }
    
    try:
        response = await session.get(url, params=params)
        return {'status': response.status_code, 'data': response.json() if response.status_code == 200 else response.text}
    except Exception as e:
        return {'status': 0, 'error': str(e)}


def parse_tweets(data: dict, keyword: str) -> list:
    """Parse tweets from GraphQL response."""
    tweets = []
    
    try:
        # Navigate the complex GraphQL response structure
        instructions = data.get('data', {}).get('search_by_raw_query', {}).get('timeline', {}).get('instructions', [])
        
        for instruction in instructions:
            entries = instruction.get('entries', [])
            
            for entry in entries:
                content = entry.get('content', {})
                item_content = content.get('itemContent', {})
                tweet_results = item_content.get('tweet_results', {})
                result = tweet_results.get('result', {})
                
                if not result or result.get('__typename') == 'TweetTombstone':
                    continue
                
                tweet = result.get('legacy', {})
                user = result.get('core', {}).get('user_results', {}).get('result', {}).get('legacy', {})
                metrics = tweet.get('public_metrics', {})
                
                tweets.append({
                    'tweet_id': result.get('rest_id', ''),
                    'text': tweet.get('full_text', ''),
                    'created_at': tweet.get('created_at', ''),
                    'user_id': tweet.get('user_id', ''),
                    'username': user.get('screen_name', ''),
                    'name': user.get('name', ''),
                    'followers': user.get('followers_count', 0),
                    'verified': 1 if user.get('verified') else 0,
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'keyword': keyword,
                })
    
    except Exception as e:
        print(f"    Parse error: {e}")
    
    return tweets


async def scrape_keyword(keyword: str, seen_ids: set) -> list:
    """Scrape tweets for a keyword."""
    all_tweets = []
    cursor = None
    
    # Create a fresh session each time
    async with httpx.AsyncClient(timeout=30.0) as session:
        # Get guest token first
        print("    Getting guest token...")
        guest_token = await get_guest_token(session)
        
        if not guest_token:
            print("    ⚠️ Could not get guest token")
            return []
        
        # Add guest token to headers
        session.headers.update({'x-guest-token': guest_token})
        
        for page in range(MAX_PAGES):
            result = await search_tweets(session, keyword, cursor)
            
            if result['status'] == 401:
                print(f"    ⚠️ Auth expired (401)")
                # Try to get new guest token
                guest_token = await get_guest_token(session)
                if guest_token:
                    session.headers.update({'x-guest-token': guest_token})
                    continue
                break
            
            if result['status'] == 429:
                print(f"    ⚠️ Rate limited (429)")
                break
            
            if result['status'] != 200:
                print(f"    ⚠️ Error: {result.get('error', 'Unknown')[:50]}")
                break
            
            tweets = parse_tweets(result.get('data', {}), keyword)
            
            if not tweets:
                break
            
            # Deduplicate
            new_tweets = [t for t in tweets if t['tweet_id'] not in seen_ids]
            for t in new_tweets:
                seen_ids.add(t['tweet_id'])
            
            all_tweets.extend(new_tweets)
            print(f"    Page {page+1}: +{len(new_tweets)} tweets")
            
            # Find next cursor
            try:
                instructions = result['data'].get('data', {}).get('search_by_raw_query', {}).get('timeline', {}).get('instructions', [])
                for inst in instructions:
                    entries = inst.get('entries', [])
                    for entry in entries:
                        if entry.get('entryId', '').startswith('cursor-'):
                            cursor = entry.get('content', {}).get('value', '')
                            break
                    if cursor:
                        break
            except:
                pass
            
            if not cursor:
                break
            
            await asyncio.sleep(1)
    
    return all_tweets


async def main():
    """Main scraping function."""
    print("=" * 60)
    print("TWITTER GRAPHQL SCRAPER v9 - Guest Token")
    print("=" * 60)
    
    # Load cookies for initial session
    print("\n[1/4] Loading cookies...")
    all_cookies = []
    for cf in COOKIE_FILES:
        try:
            cookies = load_cookies(cf)
            all_cookies.append(cookies)
            print(f"  ✓ {cf}")
        except Exception as e:
            print(f"  ✗ {cf}: {e}")
    
    # Load existing data
    print("\n[2/4] Loading existing data...")
    existing_tweets = []
    seen_ids = set()
    
    tweets_file = OUTPUT_DIR / 'enhanced_tweets.csv'
    if tweets_file.exists():
        import csv
        with open(tweets_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_tweets.append(row)
                seen_ids.add(row.get('tweet_id', ''))
        print(f"  Loaded {len(existing_tweets)} tweets")
    
    # Scrape
    print(f"\n[3/4] Scraping {len(KEYWORDS)} keywords (guest token mode)...")
    all_tweets = list(existing_tweets)
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        tweets = await scrape_keyword(keyword, seen_ids)
        
        if tweets:
            all_tweets.extend(tweets)
            print(f"    Total: {len(tweets)} new tweets")
        else:
            print(f"    No tweets found")
        
        # Save periodically
        if (i + 1) % 3 == 0:
            save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    # Save final
    print("\n[4/4] Saving...")
    save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    # Summary
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'total_tweets': len(all_tweets),
        'keywords': KEYWORDS,
    }
    with open(OUTPUT_DIR / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"📊 Total tweets: {len(all_tweets)}")
    print(f"📁 {OUTPUT_DIR}")


def save_csv(tweets: list, filepath: Path):
    """Save tweets to CSV."""
    if not tweets:
        return
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    import csv
    fieldnames = list(tweets[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tweets)


if __name__ == '__main__':
    asyncio.run(main())
