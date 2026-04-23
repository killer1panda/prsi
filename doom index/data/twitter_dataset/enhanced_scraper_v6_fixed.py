#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper V6 FIXED - GraphQL Based

Fixed to extract authentication from browser cookies properly.
Uses the cookies to make authenticated GraphQL requests.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# ============== CONFIGURATION ==============
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

# Settings
TWEETS_PER_QUERY = 100
MAX_PAGES = 10
DELAY_BETWEEN_REQUESTS = 1.0
DELAY_BETWEEN_KEYWORDS = 2.0


def load_cookies_dict(cookies_file: str) -> Dict[str, str]:
    """Load cookies as dict."""
    with open(cookies_file, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


async def get_fresh_bearer(session: httpx.AsyncClient) -> str:
    """Try to get a fresh bearer token from X mobile API."""
    # Try X mobile API which often has simpler auth
    url = "https://api.x.com/1.1/account/settings.json"
    
    try:
        r = await session.get(url)
        if r.status_code == 200:
            # Use the guest token from response headers
            return r.headers.get('x-guest-token', '')
    except:
        pass
    
    return None


def create_session(cookies_file: str) -> httpx.AsyncClient:
    """Create HTTP session with cookies - FIXED."""
    cookies = load_cookies_dict(cookies_file)
    
    # Build cookie string
    cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
    
    # Get CSRF token
    csrf_token = cookies.get('ct0', '')
    
    # Use a more current bearer token (this needs updating periodically)
    bearer = 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuBuFD3AmUBb0oZ8x2uq3k3V4bDK7p8EhZ5UwI0Q3PAB5K8pR3Fz0j9p5V2kT5OhC5PgFomG2bG4M3d0Q'
    
    headers = {
        'authorization': bearer,
        'x-csrf-token': csrf_token,
        'cookie': cookie_str,
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
    }
    
    return httpx.AsyncClient(
        headers=headers,
        cookies=cookies,
        timeout=30.0,
        follow_redirects=True
    )


async def search_graphql_v2(session: httpx.AsyncClient, query: str, cursor: str = None) -> Dict:
    """Search using X's mobile API (simpler auth)."""
    # Try mobile API endpoint
    url = "https://api.x.com/2/tweets/search/stream"
    
    params = {
        "query": query,
        "max_results": TWEETS_PER_QUERY,
    }
    
    if cursor:
        params["cursor"] = cursor
    
    try:
        response = await session.get(url, params=params)
        
        if response.status_code == 401:
            # Try alternative - get guest token first
            return {"error": "auth_required", "data": None}
        
        if response.status_code == 429:
            return {"error": "rate_limit", "data": None}
        
        if response.status_code != 200:
            return {"error": f"status_{response.status_code}", "data": None}
        
        return {"error": None, "data": response.json()}
        
    except Exception as e:
        return {"error": str(e), "data": None}


async def search_standard(session: httpx.AsyncClient, query: str, cursor: str = None) -> Dict:
    """Fallback to standard search using regular API."""
    url = "https://api.x.com/1.1/search/tweets.json"
    
    params = {
        "q": query,
        "count": TWEETS_PER_QUERY,
        "result_type": "recent",
    }
    
    if cursor:
        params["max_id"] = cursor
    
    try:
        response = await session.get(url, params=params)
        
        if response.status_code == 429:
            return {"error": "rate_limit", "data": None}
        
        if response.status_code != 200:
            return {"error": f"status_{response.status_code}", "data": None}
        
        return {"error": None, "data": response.json()}
        
    except Exception as e:
        return {"error": str(e), "data": None}


def extract_tweets_standard(data: Dict) -> tuple:
    """Extract tweets from standard search API."""
    tweets = []
    next_cursor = None
    
    try:
        statuses = data.get("statuses", [])
        
        for s in statuses:
            tweets.append({
                "tweet_id": str(s.get("id")),
                "text": s.get("text", ""),
                "created_at": s.get("created_at", ""),
                "user_id": str(s.get("user", {}).get("id", "")),
                "username": s.get("user", {}).get("screen_name", ""),
                "name": s.get("user", {}).get("name", ""),
                "followers": s.get("user", {}).get("followers_count", 0),
                "verified": 1 if s.get("user", {}).get("verified") else 0,
                "likes": s.get("favorite_count", 0),
                "retweets": s.get("retweet_count", 0),
                "replies": s.get("reply_count", 0),
                "hashtags": " ".join([h.get("text", "") for h in s.get("entities", {}).get("hashtags", [])]),
                "mentions": " ".join([m.get("screen_name", "") for m in s.get("entities", {}).get("user_mentions", [])]),
                "is_reply": 1 if s.get("in_reply_to_status_id") else 0,
                "is_retweet": 1 if s.get("retweeted_status") else 0,
            })
        
        # Next cursor
        if "search_metadata" in data:
            next_cursor = data["search_metadata"].get("next_results", "")
            if next_cursor:
                # Extract max_id from next_results
                import urllib.parse
                parsed = urllib.parse.parse_qs(next_results[1:])
                next_cursor = parsed.get("max_id", [None])[0]
    
    except Exception as e:
        print(f"    Parse error: {e}")
    
    return tweets, next_cursor


async def scrape_keyword_fixed(session: httpx.AsyncClient, keyword: str, seen_ids: set) -> tuple:
    """Scrape a keyword using standard API (more reliable)."""
    all_tweets = []
    cursor = None
    
    for page in range(MAX_PAGES):
        result = await search_standard(session, keyword, cursor)
        
        if result["error"]:
            if "rate_limit" in result["error"]:
                print(f"    ⚠️ Rate limited on page {page+1}")
                break
            print(f"    ⚠️ Error on page {page+1}: {result['error']}")
            break
        
        tweets, next_cursor = extract_tweets_standard(result.get("data", {}))
        
        if not tweets:
            break
        
        # Deduplicate
        new_tweets = [t for t in tweets if t["tweet_id"] not in seen_ids]
        for t in new_tweets:
            seen_ids.add(t["tweet_id"])
        
        all_tweets.extend(new_tweets)
        
        print(f"    Page {page+1}: +{len(new_tweets)} new tweets (total: {len(all_tweets)})")
        
        if not next_cursor:
            break
        
        cursor = next_cursor
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
    
    return all_tweets


async def scrape_fixed_graphql():
    """Main scraper with fixed authentication."""
    print("=" * 60)
    print("TWITTER SCRAPER V6 - FIXED AUTHENTICATION")
    print("=" * 60)
    print(f"\n⚡ Using standard API (more reliable than GraphQL)")
    
    # Create sessions
    print("\n[1/5] Creating HTTP sessions...")
    sessions = []
    for cf in COOKIES_FILES:
        try:
            session = create_session(cf)
            sessions.append(session)
            print(f"  ✓ {cf}")
        except Exception as e:
            print(f"  ✗ {cf}: {e}")
    
    if not sessions:
        print("ERROR: No sessions!")
        return
    
    # Load existing
    print("\n[2/5] Loading existing data...")
    all_tweets = []
    seen_ids = set()
    
    tweets_file = OUTPUT_DIR / 'enhanced_tweets.csv'
    if tweets_file.exists():
        import csv
        with open(tweets_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_tweets.append(row)
                seen_ids.add(row.get('tweet_id', ''))
        print(f"  Loaded {len(all_tweets)} tweets, {len(seen_ids)} unique")
    
    # Scrape
    print(f"\n[3/5] Scraping {len(KEYWORDS)} keywords...")
    session_idx = 0
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n  [{i+1}/{len(KEYWORDS)}] '{keyword}'")
        
        session = sessions[session_idx % len(sessions)]
        session_idx += 1
        
        # Retry
        for retry in range(3):
            try:
                await asyncio.sleep(DELAY_BETWEEN_KEYWORDS)
                tweets = await scrape_keyword_fixed(session, keyword, seen_ids)
                
                for t in tweets:
                    t['keyword'] = keyword
                
                all_tweets.extend(tweets)
                
                print(f"    Total: +{len(tweets)} tweets")
                break
                
            except Exception as e:
                if "429" in str(e):
                    print(f"    ⚠️ Rate limit, waiting...")
                    await asyncio.sleep(10)
                    if len(sessions) > 1:
                        session = sessions[session_idx % len(sessions)]
                        session_idx += 1
                else:
                    print(f"    ⚠️ Error: {e}")
                    break
        
        # Save every 3 keywords
        if (i + 1) % 3 == 0:
            save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    # Save
    print("\n[4/5] Saving...")
    save_csv(all_tweets, OUTPUT_DIR / 'enhanced_tweets.csv')
    
    summary = {
        'scraped_at': datetime.now().isoformat(),
        'total_tweets': len(all_tweets),
    }
    with open(OUTPUT_DIR / 'scrape_summary.json', 'w') as f:
        json.dump(summary, f)
    
    print("\n[5/5] Done!")
    print("=" * 60)
    print(f"📊 Tweets: {len(all_tweets)}")
    print(f"📁 {OUTPUT_DIR}")


def save_csv(data: List[Dict], filepath: Path):
    if not data:
        return
    import csv
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    asyncio.run(scrape_fixed_graphql())
