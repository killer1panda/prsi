"""
Twitter/X Dataset Scraper for Doom Index Project.
Collects cancellation-related data as per project requirements.
"""

import asyncio
import json
import csv
import os
from datetime import datetime
from twikit import Client as TwikitClient

# Configuration
COOKIES_FILE = "src/data/scrapers/twitter_cookies.json"
OUTPUT_DIR = "doom-index/data/twitter_dataset"

# Cancellation-related keywords from the project plan
CANCELLATION_KEYWORDS = [
    "#cancel",
    "#cancelled", 
    "#cancellation",
    "cancelled after",
    "facing backlash",
    "called out for",
    "under fire for",
    "controversy",
    "boycott",
    "petition",
    "sign petition",
    "cancel culture",
    "backlash",
    "outrage",
    "offensive"
]

# Target user accounts for doom index
TARGET_ACCOUNTS = [
    "github",
    "elonmusk",
    "potus",
    "realdonaldtrump",
    "joebiden",
    "kamalaharris",
    "cnn",
    "foxnews",
    "nbcnews",
    "washingtonpost"
]


def save_json(data, filename):
    """Save data to JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON: {filepath}")
    return filepath


def save_csv(data, filename):
    """Save data to CSV file."""
    if not data:
        return
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    fieldnames = list(data[0].keys()) if isinstance(data[0], dict) else []
    
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
    
    print(f"  ✓ CSV: {filepath}")


async def scrape_dataset():
    """Main scraping function for doom index data collection."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    
    # Load cookies
    print(f"\nLoading cookies...")
    with open(COOKIES_FILE, 'r') as f:
        browser_cookies = json.load(f)
    
    cookies = {c['name']: c['value'] for c in browser_cookies}
    client = TwikitClient(language='en-US')
    client.set_cookies(cookies)
    print("✓ Done!")
    
    # Initialize data collections
    cancellation_data = []
    timeline_data = []
    trending_data = []
    user_data = []
    
    # 1. Search for cancellation-related content
    print("\n[1/4] Searching cancellation keywords...")
    for keyword in CANCELLATION_KEYWORDS:
        try:
            print(f"  Searching: '{keyword}'...")
            results = await client.search_tweet(keyword, product="Top", count=30)
            print(f"    Found {len(results)} tweets")
            
            for tweet in results:
                cancellation_data.append({
                    "id": tweet.id,
                    "keyword": keyword,
                    "user": tweet.user.screen_name,
                    "user_name": tweet.user.name,
                    "text": tweet.text,
                    "created_at": str(tweet.created_at) if tweet.created_at else "",
                    "likes": tweet.favorite_count,
                    "retweets": tweet.retweet_count,
                    "replies": tweet.reply_count,
                    "source": "cancellation_search"
                })
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n  Total cancellation data: {len(cancellation_data)} tweets")
    
    # 2. Get home timeline
    print("\n[2/4] Fetching timeline...")
    try:
        timeline = await client.get_latest_timeline(count=100)
        print(f"  Found {len(timeline)} tweets")
        
        for tweet in timeline:
            timeline_data.append({
                "id": tweet.id,
                "user": tweet.user.screen_name,
                "user_name": tweet.user.name,
                "text": tweet.text,
                "created_at": str(tweet.created_at) if tweet.created_at else "",
                "likes": tweet.favorite_count,
                "retweets": tweet.retweet_count,
                "replies": tweet.reply_count,
                "source": "timeline"
            })
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 3. Get trending topics
    print("\n[3/4] Fetching trends...")
    try:
        trends = await client.get_trends(category="trending")
        print(f"  Found {len(trends)} trends")
        
        for trend in trends:
            trending_data.append({
                "name": trend.name,
                "domain": trend.domain_context,
                "tweets_count": trend.tweets_count,
                "category": "trending",
                "source": "trending"
            })
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 4. Get target user data
    print("\n[4/4] Fetching target accounts...")
    for username in TARGET_ACCOUNTS:
        try:
            # Get user info
            try:
                user = await client.get_user_by_screen_name(username)
                user_data.append({
                    "username": user.screen_name,
                    "name": user.name,
                    "bio": user.description if hasattr(user, 'description') else "",
                    "followers": user.followers_count,
                    "following": user.friends_count,
                    "tweets_count": user.statuses_count,
                    "verified": user.verified if hasattr(user, 'verified') else False,
                    "created_at": str(user.created_at) if hasattr(user, 'created_at') and user.created_at else "",
                    "source": "user_info"
                })
            except:
                pass
            
            # Get user tweets
            tweets = await client.get_user_tweets(username, tweet_type="Tweets", count=20)
            print(f"  @{username}: {len(tweets)} tweets")
            
            for tweet in tweets:
                timeline_data.append({
                    "id": tweet.id,
                    "user": username,
                    "text": tweet.text,
                    "created_at": str(tweet.created_at) if tweet.created_at else "",
                    "likes": tweet.favorite_count,
                    "retweets": tweet.retweet_count,
                    "replies": tweet.reply_count,
                    "source": "user_tweets"
                })
                
        except Exception as e:
            print(f"  ✗ @{username}: {e}")
    
    # Save all data
    print("\n[Saving...]")
    
    # Cancellation data (main focus for doom index)
    save_json(cancellation_data, "cancellation_events.json")
    save_csv(cancellation_data, "cancellation_events.csv")
    
    # Timeline data
    save_json(timeline_data, "timeline_tweets.json")
    save_csv(timeline_data, "timeline_tweets.csv")
    
    # Trending data
    save_json(trending_data, "trending_topics.json")
    save_csv(trending_data, "trending_topics.csv")
    
    # User data
    save_json(user_data, "target_users.json")
    save_csv(user_data, "target_users.csv")
    
    # Complete dataset
    complete = {
        "scraped_at": datetime.now().isoformat(),
        "project": "Doom Index - Predictive Social Doom Index",
        "cancellation_events": cancellation_data,
        "timeline_tweets": timeline_data,
        "trending_topics": trending_data,
        "target_users": user_data,
        "total_records": len(cancellation_data) + len(timeline_data) + len(trending_data) + len(user_data)
    }
    save_json(complete, "complete_dataset.json")
    
    # Summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETE!")
    print("="*60)
    print(f"\nDoom Index Dataset Summary:")
    print(f"  - Cancellation Events: {len(cancellation_data)}")
    print(f"  - Timeline Tweets: {len(timeline_data)}")
    print(f"  - Trending Topics: {len(trending_data)}")
    print(f"  - Target Users: {len(user_data)}")
    print(f"\n  TOTAL RECORDS: {complete['total_records']}")
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    
    return complete


if __name__ == "__main__":
    asyncio.run(scrape_dataset())
