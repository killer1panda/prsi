"""
Simple script to run twikit scraper with provided credentials.

Usage:
    python run_twitter_scraper.py

Credentials provided:
- Username: doomlord14686
- Email: vaasha038@gmail.com
- Password: Hesoyam1@
"""

import asyncio
import json
import os
from datetime import datetime
from twikit import Client

# Credentials
EMAIL = "vaasha038@gmail.com"
USERNAME = "doomlord14686"
PASSWORD = "Hesoyam1@"

# Session file to save/login from
SESSION_FILE = "twitter_session.json"


async def login_and_save_session():
    """Login to Twitter and save session."""
    print("Initializing Twikit client...")
    # Using a custom user agent and proxy to bypass Cloudflare
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    # Using a proxy from the proxy list
    proxy = 'http://95.217.195.146:9999'
    client = Client(language='en-US', user_agent=user_agent, proxy=proxy)
    
    try:
        print(f"Logging in with username: {USERNAME}, email: {EMAIL}")
        await client.login(
            auth_info_1=EMAIL,
            auth_info_2=USERNAME,
            password=PASSWORD
        )
        print("Login successful!")
        
        # Save session
        client.save_session(SESSION_FILE)
        print(f"Session saved to {SESSION_FILE}")
        
        return client
    except Exception as e:
        print(f"Login failed: {e}")
        return None


async def load_existing_session():
    """Try to load an existing session."""
    if os.path.exists(SESSION_FILE):
        print(f"Loading existing session from {SESSION_FILE}...")
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        proxy = 'http://95.217.195.146:9999'
        client = Client(language='en-US', user_agent=user_agent, proxy=proxy)
        try:
            client.load_session(SESSION_FILE)
            print("Session loaded successfully!")
            return client
        except Exception as e:
            print(f"Failed to load session: {e}")
            return None
    return None


async def get_user_info(client, username):
    """Get user information."""
    print(f"\nFetching user info for: {username}")
    try:
        user = await client.get_user(username)
        print(f"User found: {user.name} (@{user.screen_name})")
        print(f"  - Followers: {user.followers_count}")
        print(f"  - Following: {user.friends_count}")
        print(f"  - Tweets: {user.statuses_count}")
        print(f"  - Verified: {user.verified}")
        return user
    except Exception as e:
        print(f"Error fetching user info: {e}")
        return None


async def get_user_tweets(client, username, count=10):
    """Get user's recent tweets."""
    print(f"\nFetching recent tweets for: {username}")
    try:
        tweets = await client.get_user_tweets(username, count)
        print(f"Found {len(tweets)} tweets:")
        for i, tweet in enumerate(tweets[:count], 1):
            print(f"\n  Tweet {i}:")
            print(f"    {tweet.text[:200]}...")
            print(f"    Likes: {tweet.favorite_count} | Retweets: {tweet.retweet_count}")
            print(f"    Created: {tweet.created_at}")
        return tweets
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []


async def search_tweets(client, query, count=10):
    """Search for tweets."""
    print(f"\nSearching for: {query}")
    try:
        tweets = await client.search(query, count)
        print(f"Found {len(tweets)} tweets:")
        for i, tweet in enumerate(tweets[:count], 1):
            print(f"\n  Tweet {i}:")
            print(f"    {tweet.text[:200]}...")
            print(f"    User: @{tweet.user.screen_name}")
            print(f"    Likes: {tweet.favorite_count} | Retweets: {tweet.retweet_count}")
        return tweets
    except Exception as e:
        print(f"Error searching tweets: {e}")
        return []


async def get_trending(client):
    """Get trending topics."""
    print("\nFetching trending topics...")
    try:
        trends = await client.get_trending()
        print(f"Found {len(trends)} trending topics:")
        for i, trend in enumerate(trends[:10], 1):
            print(f"  {i}. {trend.name} - {trend.tweet_count} tweets")
        return trends
    except Exception as e:
        print(f"Error fetching trends: {e}")
        return []


async def main():
    """Main function."""
    print("=" * 50)
    print("Twitter/X Scraper using Twikit")
    print("=" * 50)
    
    # Try to load existing session first
    client = await load_existing_session()
    
    if not client:
        # Login with credentials
        client = await login_and_save_session()
    
    if not client:
        print("Failed to authenticate. Please check credentials.")
        return
    
    # Demonstrate various features
    print("\n" + "=" * 50)
    print("Scraping Data")
    print("=" * 50)
    
    # Get user info
    user = await get_user_info(client, USERNAME)
    
    # Get user tweets
    if user:
        await get_user_tweets(client, USERNAME, count=5)
    
    # Search for something
    await search_tweets(client, "Python programming", count=5)
    
    # Get trending
    await get_trending(client)
    
    print("\n" + "=" * 50)
    print("Scraping complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
