#!/usr/bin/env python3
"""Enhanced Twitter/X Scraper V7 - RESEARCH GRADE

The ultimate scraper for large-scale dataset collection:
- GraphQL API (fast)
- Async workers (parallel)
- SQLite storage (efficient)
- Cursor pagination
- Rate limit handling
- Resume capability

Target: 100k-300k tweets per day with proper setup.
"""

import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from queue import Queue
import threading

import httpx

# ============== CONFIGURATION ==============
COOKIES_FILES = [
    'doom-index/src/data/scrapers/twitter_cookies.json',
    'doom-index/src/data/scrapers/twitter_cookies2.json',
]

OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
DB_FILE = OUTPUT_DIR / 'tweets.db'
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
WORKERS = 5
DELAY = 1.0


def init_db():
    """Initialize SQLite database."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    
    # Tweets table
    c.execute('''CREATE TABLE IF NOT EXISTS tweets (
        id TEXT PRIMARY KEY,
        keyword TEXT,
        text TEXT,
        created_at TEXT,
        user_id TEXT,
        username TEXT,
        name TEXT,
        followers INTEGER,
        verified INTEGER,
        likes INTEGER,
        retweets INTEGER,
        replies INTEGER,
        hashtags TEXT,
        mentions TEXT,
        is_reply INTEGER,
        is_retweet INTEGER,
        scraped_at TEXT
    )''')
    
    # Replies table
    c.execute('''CREATE TABLE IF NOT EXISTS replies (
        id TEXT PRIMARY KEY,
        parent_id TEXT,
        keyword TEXT,
        text TEXT,
        created_at TEXT,
        username TEXT,
        name TEXT,
        likes INTEGER,
        retweets INTEGER,
        scraped_at TEXT
    )''')
    
    # Progress table
    c.execute('''CREATE TABLE IF NOT EXISTS progress (
        keyword TEXT PRIMARY KEY,
        cursor TEXT,
        pages_done INTEGER,
        last_scraped TEXT
    )''')
    
    conn.commit()
    conn.close()
    print(f"  ✓ Database initialized: {DB_FILE}")


def load_cookies(cookies_file: str) -> Dict[str, str]:
    with open(cookies_file, 'r') as f:
        cookies_list = json.load(f)
    return {c['name']: c['value'] for c in cookies_list}


def create_session(cookies_file: str) -> httpx.AsyncClient:
    cookies = load_cookies(cookies_file)
    cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
    
    headers = {
        'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuBuFD3AmUBb0oZ8x2uq3k3V' + '4bDK7p8EhZ5UwI0Q3PAB5K8pR3Fz0j9p5V2kT5OhC5PgFomG2bG4M3d0Q',
        'x-csrf-token': cookies.get('ct0', ''),
        'cookie': cookie_str,
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }
    
    return httpx.AsyncClient(headers=headers, cookies=cookies, timeout=30.0)


async def search_graphql(session: httpx.AsyncClient, query: str, cursor: str = None) -> Dict:
    """Search using GraphQL."""
    url = "https://twitter.com/i/api/graphql/9d1M63g9Ux2GC3xD7k1xMw/SearchTimeline"
    
    variables = {"rawQuery": query, "count": TWEETS_PER_QUERY, "product": "Latest"}
    if cursor:
        variables["cursor"] = cursor
    
    params = {"variables": json.dumps(variables), "features": json.dumps({
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    })}
    
    try:
        r = await session.get(url, params=params)
        if r.status_code == 429:
            return {"error": "rate_limit", "data": None}
        if r.status_code != 200:
            return {"error": f"status_{r.status_code}", "data": None}
        return {"error": None, "data": r.json()}
    except Exception as e:
        return {"error": str(e), "data": None}


def extract_tweets(data: Dict) -> tuple:
    """Extract tweets from GraphQL response."""
    tweets = []
    cursor = None
    
    try:
        instructions = data.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {}).get("instructions", [])
        
        for inst in instructions:
            if inst.get("type") != "TimelineAddEntries":
                continue
            
            for entry in inst.get("entries", []):
                entry_id = entry.get("entryId", "")
                
                if "tweet-" in entry_id or "retweet-" in entry_id:
                    try:
                        result = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result", {})
                        legacy = result.get("legacy", {})
                        if not legacy:
                            continue
                        
                        user = result.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
                        
                        tweets.append({
                            "id": result.get("rest_id", ""),
                            "text": legacy.get("full_text", ""),
                            "created_at": legacy.get("created_at", ""),
                            "user_id": user.get("id_str", ""),
                            "username": user.get("screen_name", ""),
                            "name": user.get("name", ""),
                            "followers": user.get("followers_count", 0),
                            "verified": 1 if result.get("core", {}).get("user_results", {}).get("result", {}).get("is_blue_verified") else 0,
                            "likes": legacy.get("favorite_count", 0),
                            "retweets": legacy.get("retweet_count", 0),
                            "replies": legacy.get("reply_count", 0),
                            "hashtags": " ".join([h["text"] for h in legacy.get("entities", {}).get("hashtags", [])]),
                            "mentions": " ".join([m["screen_name"] for m in legacy.get("entities", {}).get("user_mentions", [])]),
                            "is_reply": 1 if legacy.get("in_reply_to_status_id_str") else 0,
                            "is_retweet": 1 if legacy.get("retweeted_status_result") else 0,
                        })
                    except:
                        continue
                
                elif "cursor-bottom" in entry_id:
                    cursor = entry.get("content", {}).get("value", "")
    
    except Exception as e:
        pass
    
    return tweets, cursor


def save_to_db(tweets: List[Dict], keyword: str):
    """Save tweets to SQLite."""
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    
    now = datetime.now().isoformat()
    
    for t in tweets:
        try:
            c.execute('''INSERT OR IGNORE INTO tweets 
                (id, keyword, text, created_at, user_id, username, name, followers, verified, 
                 likes, retweets, replies, hashtags, mentions, is_reply, is_retweet, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (t['id'], keyword, t['text'], t['created_at'], t['user_id'], t['username'], 
                 t['name'], t['followers'], t['verified'], t['likes'], t['retweets'],
                 t['replies'], t['hashtags'], t['mentions'], t['is_reply'], t['is_retweet'], now))
        except:
            pass
    
    conn.commit()
    conn.close()


def get_progress(keyword: str) -> tuple:
    """Get scraping progress for a keyword."""
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("SELECT cursor, pages_done FROM progress WHERE keyword = ?", (keyword,))
    row = c.fetchone()
    conn.close()
    return (row[0], row[1]) if row else (None, 0)


def save_progress(keyword: str, cursor: str, pages: int):
    """Save scraping progress."""
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO progress (keyword, cursor, pages_done, last_scraped)
                VALUES (?, ?, ?, ?)''',
                (keyword, cursor, pages, datetime.now().isoformat()))
    conn.commit()
    conn.close()


async def worker(worker_id: int, keyword_queue: Queue, sessions: List, results: dict):
    """Worker coroutine for parallel scraping."""
    session_idx = worker_id
    
    while True:
        keyword = keyword_queue.get()
        if keyword is None:
            break
        
        print(f"  [Worker {worker_id}] Processing: {keyword}")
        
        session = sessions[session_idx % len(sessions)]
        session_idx += 1
        
        cursor, pages_done = get_progress(keyword)
        
        all_tweets = []
        
        for page in range(pages_done, MAX_PAGES):
            result = await search_graphql(session, keyword, cursor)
            
            if result["error"]:
                if "rate_limit" in result["error"]:
                    print(f"    [Worker {worker_id}] Rate limited!")
                    await asyncio.sleep(30)
                    continue
                break
            
            tweets, next_cursor = extract_tweets(result["data"])
            
            if not tweets:
                break
            
            # Deduplicate within batch
            new_tweets = [t for t in tweets if t['id'] not in results['seen_ids']]
            for t in new_tweets:
                results['seen_ids'].add(t['id'])
            
            all_tweets.extend(new_tweets)
            cursor = next_cursor
            
            print(f"    [Worker {worker_id}] Page {page+1}: +{len(new_tweets)} tweets")
            
            await asyncio.sleep(DELAY)
            
            # Save progress
            save_progress(keyword, cursor, page + 1)
        
        # Save to DB
        if all_tweets:
            save_to_db(all_tweets, keyword)
            results['total_tweets'] += len(all_tweets)
            print(f"    [Worker {worker_id}] Saved {len(all_tweets)} tweets for '{keyword}'")
        
        keyword_queue.task_done()


async def scrape_research_grade():
    """Main research-grade scraper."""
    print("=" * 60)
    print("TWITTER SCRAPER V7 - RESEARCH GRADE")
    print("=" * 60)
    print(f"\n⚡ Features:")
    print(f"   - SQLite database (efficient storage)")
    print(f"   - {WORKERS} parallel workers")
    print(f"   - GraphQL API (fast)")
    print(f"   - Resume capability")
    print(f"   - {MAX_PAGES} pages per keyword")
    
    # Init DB
    print("\n[1/4] Initializing database...")
    init_db()
    
    # Create sessions
    print("\n[2/4] Creating sessions...")
    sessions = []
    for cf in COOKIES_FILES:
        try:
            sessions.append(create_session(cf))
            print(f"  ✓ {cf}")
        except Exception as e:
            print(f"  ✗ {cf}: {e}")
    
    if not sessions:
        print("ERROR: No sessions!")
        return
    
    # Load seen IDs
    print("\n[3/4] Loading existing data...")
    seen_ids = set()
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("SELECT id FROM tweets")
    for row in c.fetchall():
        seen_ids.add(row[0])
    conn.close()
    print(f"  Loaded {len(seen_ids)} existing tweet IDs")
    
    # Queue keywords
    keyword_queue = Queue()
    for kw in KEYWORDS:
        keyword_queue.put(kw)
    
    # Results tracker
    results = {'seen_ids': seen_ids, 'total_tweets': len(seen_ids)}
    
    # Start workers
    print(f"\n[4/4] Starting {WORKERS} workers...")
    workers = [asyncio.create_task(worker(i, keyword_queue, sessions, results)) for i in range(WORKERS)]
    
    # Wait for completion
    keyword_queue.join()
    
    # Stop workers
    for _ in range(WORKERS):
        keyword_queue.put(None)
    await asyncio.gather(*workers)
    
    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   - Total tweets in DB: {results['total_tweets']}")
    print(f"📁 Database: {DB_FILE}")


if __name__ == '__main__':
    asyncio.run(scrape_research_grade())
