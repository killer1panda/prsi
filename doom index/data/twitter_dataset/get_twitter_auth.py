#!/usr/bin/env python3
"""
Twitter GraphQL Scraper with Fresh Authentication
Uses Playwright to login and extract fresh tokens for GraphQL queries.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

# ============== CONFIGURATION ==============
OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Twitter credentials
TWITTER_USERNAME = "doomlord14686"
TWITTER_EMAIL = "vaasha038@gmail.com"
TWITTER_PASSWORD = "Hesoyam1@"

# Keywords to search
KEYWORDS = [
    '#cancel', '#cancelled', '#cancellation', 'cancelled after', 
    'facing backlash', 'called out for', 'under fire for', 
    'controversy', 'boycott', 'petition', 'cancel culture',
    'backlash', 'outrage'
]

# GraphQL query for search (using Twitter's actual query)
SEARCH_QUERY = '''
query($query: String!, $count: Int!, $cursor: String) {
  search_by_raw_query(query: $query, count: $count, cursor: $cursor) {
    timeline {
      instructions {
        type
        entries {
          entryId
          sortIndex
          content {
            entryType
            __typename
            itemContent {
              itemType
              __typename
              tweet_results {
                result {
                  ... on Tweet {
                    rest_id
                    legacy {
                      full_text
                      created_at
                      favorite_count
                      retweet_count
                      reply_count
                      user_id
                      is_nullsafe
                    }
                    core {
                      user_results {
                        result {
                          legacy {
                            screen_name
                            name
                            followers_count
                            verified
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
'''

# Simpler search query that works better
SEARCH_QUERY_V2 = '''
query SearchTimeline($query: String!, $count: Int!, $cursor: String) {
  search_raw(query: $query, count: $count, cursor: $cursor) {
    results {
      timeline {
        instructions {
          type
          entries {
            entryId
            content {
              entryType
              ... on TimelineTimelineItem {
                itemContent {
                  itemType
                  tweet_results {
                    result {
                      ... on TweetTombstone {
                        tombstone {
                          text
                        }
                      }
                      ... on Tweet {
                        rest_id
                        legacy {
                          full_text
                          created_at
                          favorite_count
                          retweet_count
                          reply_count
                        }
                        core {
                          user_results {
                            result {
                              legacy {
                                screen_name
                                name
                                followers_count
                                verified
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
'''


class TwitterAuthenticator:
    """Authenticate to Twitter and extract fresh tokens."""
    
    def __init__(self):
        self.context = None
        self.page = None
        self.auth_token = None
        self.bearer_token = None
        self.ct0 = None
        self.cookies = {}
        
    async def login(self):
        """Login to Twitter using Playwright."""
        print("[AUTH] Starting browser...")
        
        async with async_playwright() as p:
            # Create browser with stealth settings
            browser = await p.chromium.launch(
                headless=False,  # Need to see the browser
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox'
                ]
            )
            
            # Create context with realistic settings
            self.context = await browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York'
            )
            
            self.page = await self.context.new_page()
            
            # Enable request interception to capture tokens
            self.bearer_token = None
            
            async def handle_response(response):
                """Capture bearer token from responses."""
                if 'api.x.com' in response.url:
                    auth = response.headers.get('authorization', '')
                    if auth and auth.startswith('Bearer'):
                        self.bearer_token = auth
                
                # Also check for cookies in response
                if 'twitter.com' in response.url:
                    try:
                        cookies = await self.context.cookies()
                        for c in cookies:
                            self.cookies[c['name']] = c['value']
                            if c['name'] == 'ct0':
                                self.ct0 = c['value']
                    except:
                        pass
            
            self.page.on('response', handle_response)
            
            print("[AUTH] Opening Twitter...")
            await self.page.goto('https://twitter.com/i/flow/login', wait_until='networkidle')
            
            # Enter username
            print("[AUTH] Entering username...")
            await self.page.wait_for_timeout(1000)
            
            # Try different selectors for username field
            try:
                await self.page.fill('input[autocomplete="username"]', TWITTER_USERNAME)
            except:
                try:
                    await self.page.fill('input[type="text"]', TWITTER_USERNAME)
                except:
                    await self.page.fill('input[name="text"]', TWITTER_USERNAME)
            
            await self.page.wait_for_timeout(500)
            
            # Click next
            try:
                await self.page.click('button:has-text("Next")')
            except:
                await self.page.keyboard.press('Enter')
            
            await self.page.wait_for_timeout(1500)
            
            # Enter email if prompted
            print("[AUTH] Entering email...")
            try:
                await self.page.fill('input[type="text"]', TWITTER_EMAIL)
                await self.page.wait_for_timeout(500)
                try:
                    await self.page.click('button:has-text("Next")')
                except:
                    await self.page.keyboard.press('Enter')
                await self.page.wait_for_timeout(1500)
            except Exception as e:
                print(f"[AUTH] Email step skipped: {e}")
            
            # Enter password
            print("[AUTH] Entering password...")
            try:
                await self.page.fill('input[type="password"]', TWITTER_PASSWORD)
                await self.page.wait_for_timeout(500)
                
                # Click login
                try:
                    await self.page.click('button:has-text("Log in")')
                except:
                    await self.page.keyboard.press('Enter')
                
                print("[AUTH] Waiting for login...")
                await self.page.wait_for_timeout(5000)
            except Exception as e:
                print(f"[AUTH] Password error: {e}")
            
            # Wait for home page
            try:
                await self.page.wait_for_url('https://twitter.com/home', timeout=10000)
                print("[AUTH] ✓ Logged in successfully!")
            except:
                # Check if we're logged in
                current_url = self.page.url
                print(f"[AUTH] Current URL: {current_url}")
                if 'home' not in current_url:
                    print("[AUTH] ⚠️ May not be fully logged in")
            
            # Extract all cookies
            print("[AUTH] Extracting authentication tokens...")
            all_cookies = await self.context.cookies()
            
            for cookie in all_cookies:
                self.cookies[cookie['name']] = cookie['value']
                print(f"  - {cookie['name']}: {cookie['value'][:20]}...")
                
                if cookie['name'] == 'ct0':
                    self.ct0 = cookie['value']
            
            # Try to get bearer token by making a request
            print("[AUTH] Getting bearer token...")
            try:
                await self.page.goto('https://twitter.com/explore', wait_until='networkidle')
                await self.page.wait_for_timeout(2000)
            except:
                pass
            
            # Save auth data
            self.save_auth()
            
            await browser.close()
            
        return self.cookies, self.bearer_token
    
    def save_auth(self):
        """Save authentication data."""
        auth_data = {
            'cookies': self.cookies,
            'bearer_token': self.bearer_token,
            'ct0': self.ct0,
            'timestamp': datetime.now().isoformat()
        }
        
        auth_file = OUTPUT_DIR / 'fresh_auth.json'
        with open(auth_file, 'w') as f:
            json.dump(auth_data, f, indent=2)
        
        print(f"[AUTH] Saved to {auth_file}")
        print(f"[AUTH] Bearer: {self.bearer_token}")
        print(f"[AUTH] CT0: {self.ct0[:20] if self.ct0 else 'None'}...")


async def main():
    """Main function."""
    print("=" * 60)
    print("TWITTER AUTHENTICATOR - GraphQL Version")
    print("=" * 60)
    print("\nThis will:")
    print("1. Open a browser")
    print("2. Log into Twitter with your credentials")
    print("3. Extract fresh authentication tokens")
    print("4. Save them for GraphQL scraping\n")
    
    auth = TwitterAuthenticator()
    cookies, bearer = await auth.login()
    
    print("\n" + "=" * 60)
    print("AUTHENTICATION COMPLETE")
    print("=" * 60)
    print(f"Cookies: {len(cookies)}")
    print(f"Bearer: {bearer[:30] if bearer else 'Not captured'}...")
    print(f"CT0: {auth.ct0[:20] if auth.ct0 else 'None'}...")
    print(f"\nSaved to: {OUTPUT_DIR / 'fresh_auth.json'}")
    print("\nNow you can use these tokens with the GraphQL scraper!")


if __name__ == '__main__':
    asyncio.run(main())
