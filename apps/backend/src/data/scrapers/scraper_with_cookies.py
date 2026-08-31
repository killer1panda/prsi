"""
Twitter/X Scraper using Twikit with cookies.
Saves data to JSON files.
"""

import asyncio
import json
import os
from datetime import datetime
from twikit import Client

COOKIES_FILE = "twitter_cookies.json"
OUTPUT_DIR = "scraped_data"


async def main():
    """Main scraping function."""
    print("=" * 60)
    print("Twitter/X Scraper with Twikit")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(COOKIES_FILE):
        print(f"Cookie file not found: {COOKIES_FILE}")
        return
    
    print(f"Loading cookies from {COOKIES_FILE}...")
    
    try:
        # Load cookies from browser export
        with open(COOKIES_FILE, 'r') as f:
            browser_cookies = json.load(f)
        
        # Convert to Twikit format (dict with name as key)
        cookies = {c['name']: c['value'] for c in browser_cookies}
        
        # Create client and set cookies
        client = Client(language='en-US')
        client.set_cookies(cookies)
        
        print("✓ Cookies loaded!")
        
        all_data = {
            "scraped_at": datetime.now().isoformat(),
            "timeline_tweets": [],
            "search_results": [],
            "trending_topics": []
        }
        
        # Get timeline (home feed)
        print("\n1. Fetching home timeline...")
        try:
            timeline = await client.get_latest_timeline(count=100)
            print(f"   ✓ Found {len(timeline)} tweets")
            
            for tweet in timeline:
                all_data["timeline_tweets"].append({
                    "id": tweet.id,
                    "user": tweet.user.screen_name,
                    "user_name": tweet.user.name,
                    "text": tweet.text,
                    "created_at": str(tweet.created_at) if tweet.created_at else None,
                    "likes": tweet.favorite_count,
                    "retweets": tweet.retweet_count,
                    "replies": tweet.reply_count,
                    "is_retweet": tweet.retweeted_tweet is not None if hasattr(tweet, 'retweeted_tweet') else False
                })
        except Exception as e:
            print(f"   ✗ Timeline failed: {e}")
        
        # Search for tweets
        print("\n2. Searching for 'Python'...")
        try:
            search_results = await client.search_tweet('Python', product='Top', count=50)
            print(f"   ✓ Found {len(search_results)} tweets")
            
            for tweet in search_results:
                all_data["search_results"].append({
                    "id": tweet.id,
                    "user": tweet.user.screen_name,
                    "user_name": tweet.user.name,
                    "text": tweet.text,
                    "created_at": str(tweet.created_at) if tweet.created_at else None,
                    "likes": tweet.favorite_count,
                    "retweets": tweet.retweet_count
                })
        except Exception as e:
            print(f"   ✗ Search failed: {e}")
        
        # Get trending topics
        print("\n3. Fetching trending topics...")
        try:
            trends = await client.get_trends(category='trending', count=50)
            print(f"   ✓ Found {len(trends)} trends")
            
            for trend in trends:
                all_data["trending_topics"].append({
                    "name": trend.name,
                    "domain": trend.domain_context,
                    "tweets_count": trend.tweets_count
                })
        except Exception as e:
            print(f"   ✗ Trending failed: {e}")
        
        # Save all data to JSON
        output_file = os.path.join(OUTPUT_DIR, "twitter_data.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Data saved to: {output_file}")
        
        # Also save timeline separately
        timeline_file = os.path.join(OUTPUT_DIR, "timeline_tweets.json")
        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(all_data["timeline_tweets"], f, indent=2, ensure_ascii=False)
        print(f"✓ Timeline saved to: {timeline_file}")
        
        # Save search results
        search_file = os.path.join(OUTPUT_DIR, "search_python.json")
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(all_data["search_results"], f, indent=2, ensure_ascii=False)
        print(f"✓ Search results saved to: {search_file}")
        
        # Save trending
        trending_file = os.path.join(OUTPUT_DIR, "trending_topics.json")
        with open(trending_file, 'w', encoding='utf-8') as f:
            json.dump(all_data["trending_topics"], f, indent=2, ensure_ascii=False)
        print(f"✓ Trending saved to: {trending_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE!")
        print("=" * 60)
        print(f"Timeline tweets: {len(all_data['timeline_tweets'])}")
        print(f"Search results: {len(all_data['search_results'])}")
        print(f"Trending topics: {len(all_data['trending_topics'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
