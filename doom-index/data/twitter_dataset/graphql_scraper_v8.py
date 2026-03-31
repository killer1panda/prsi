#!/usr/bin/env python3
"""
Twitter GraphQL Scraper - Using Mobile API
The mobile API has simpler authentication requirements.
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
    'backlash', 'outrage', 'offensive', 'scandal'
]

MAX_TWEETS = 10000
TWEETS_PER_PAGE = 50
MAX_PAGES = 20


def load_cookies(path: str) -> dict:
    """Load cookies from JSON file."""
    with open(path, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def create_session(cookies: dict) -> httpx.AsyncClient:
    """Create HTTP session with proper headers."""
    cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
    csrf = cookies.get('ct0', '')
    
    # Current bearer token (needs periodic updates)
    bearer = 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuBuFD3AmUBb0oZ8x2uq3k3V4bDK7p8EhZ5UwI0Q3PAB5K8pR3Fz0j9p5V2kT5OhC5PgFomG2bG4M3d0Q'
    
    headers = {
        'authorization': bearer,
        'x-csrf-token': csrf,
        'cookie': cookie_str,
        'user-agent': 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.230 Mobile Safari/537.36',
        'content-type': 'application/json',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'x-twitter-active-user': 'yes',
        'x-twitter-client-language': 'en',
    }
    
    return httpx.AsyncClient(
        headers=headers,
        cookies=cookies,
        timeout=30.0,
        follow_redirects=True
    )


async def search_tweets_mobile(session: httpx.AsyncClient, keyword: str, cursor: str = None) -> dict:
    """Search using Twitter's mobile API endpoint."""
    
    # Mobile API endpoint
    url = "https://api.x.com/2/tweets/search/stream/relevance"
    
    params = {
        'query': keyword,
        'tweet.fields': 'created_at,author_id,public_metrics,text',
        'expansions': 'author_id',
        'user.fields': 'name,username,public_metrics,verified',
        'max_results': TWEETS_PER_PAGE,
    }
    
    if cursor:
        params['cursor'] = cursor
    
    try:
        response = await session.get(url, params=params)
        return {'status': response.status_code, 'data': response.json() if response.status_code == 200 else response.text}
    except Exception as e:
        return {'status': 0, 'error': str(e)}


async def search_tweets_graphql(session: httpx.AsyncClient, keyword: str, cursor: str = None) -> dict:
    """Search using Twitter's GraphQL API."""
    
    # GraphQL search endpoint
    url = "https://api.x.com/graphql/1wjM72S3U5JIsD7Y8G6Qw/SearchTimeline"
    
    variables = {
        'rawQuery': keyword,
        'count': TWEETS_PER_PAGE,
        'cursor': cursor,
        'querySource': 'typed_query',
    }
    
    params = {
        'variables': json.dumps(variables),
    }
    
    try:
        response = await session.get(url, params=params)
        
        if response.status_code == 200:
            return {'status': 200, 'data': response.json()}
        elif response.status_code == 401:
            return {'status': 401, 'error': 'Unauthorized - token expired'}
        elif response.status_code == 429:
            return {'status': 429, 'error': 'Rate limited'}
        else:
            return {'status': response.status_code, 'error': response.text[:200]}
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


async def scrape_keyword(session: httpx.AsyncClient, keyword: str, seen_ids: set) -> list:
    """Scrape tweets for a keyword."""
    all_tweets = []
    cursor = None
    
    for page in range(MAX_PAGES):
        result = await search_tweets_graphql(session, keyword, cursor)
        
        if result['status'] == 401:
            print(f"    ⚠️ Auth expired (401)")
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
    print("TWITTER GRAPHQL SCRAPER")
    print("=" * 60)
    
    # Load cookies
    print("\n[1/4] Loading cookies...")
    all_cookies = []
    for cf in COOKIE_FILES:
        try:
            cookies = load_cookies(cf)
            all_cookies.append(cookies)
            print(f"  ✓ {cf}")
        except Exception as e:
            print(f"  ✗ {cf}: {e}")
    
    if not all_cookies:
        print("ERROR: No cookies loaded!")
        return
    
    # Create sessions
    sessions = [create_session(c) for c in all_cookies]
    print(f"  Created {len(sessions)} sessions")
    
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
    print(f"\n[3/4] Scraping {len(KEYWORDS)} keywords...")
    all_tweets = list(existing_tweets)
    
    session_idx = 0
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        session = sessions[session_idx % len(sessions)]
        session_idx += 1
        
        # Try with backoff
        for retry in range(3):
            tweets = await scrape_keyword(session, keyword, seen_ids)
            
            if tweets:
                for t in tweets:
                    t['keyword'] = keyword
                all_tweets.extend(tweets)
                print(f"    Total: {len(tweets)} new tweets")
                break
            
            if retry < 2:
                print(f"    Retry {retry+1}...")
                await asyncio.sleep(3)
        
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
