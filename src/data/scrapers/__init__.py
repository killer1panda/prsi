"""Data scrapers for social media platforms."""

from src.data.scrapers.twitter_scraper import TwitterScraper, create_twitter_scraper
from src.data.scrapers.reddit_scraper import RedditScraper, create_reddit_scraper
from src.data.scrapers.instagram_scraper import InstagramScraper, create_instagram_scraper

__all__ = [
    "TwitterScraper",
    "create_twitter_scraper",
    "RedditScraper",
    "create_reddit_scraper",
    "InstagramScraper",
    "create_instagram_scraper",
]
