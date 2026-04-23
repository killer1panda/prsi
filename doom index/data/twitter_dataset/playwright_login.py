"""
Use Playwright to login to Twitter/X and extract cookies.
"""

import asyncio
import json
from playwright.async_api import async_playwright

# Credentials
EMAIL = "vaasha038@gmail.com"
USERNAME = "doomlord14686"
PASSWORD = "Hesoyam1@"

COOKIES_FILE = "twitter_cookies.json"


async def login_and_get_cookies():
    """Login to Twitter and get cookies using Playwright."""
    print("Starting Playwright...")
    
    async with async_playwright() as p:
        # Launch browser (headless=False to see what's happening)
        browser = await p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        # Create context with stealth settings
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        
        # Add stealth scripts
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        
        try:
            print("Opening Twitter login page...")
            await page.goto("https://twitter.com/i/flow/login", wait_until="networkidle")
            
            # Wait for login form
            await asyncio.sleep(5)
            
            print("Entering username...")
            # Try to find username input
            username_input = await page.wait_for_selector(
                "input[autocomplete='username']", 
                timeout=10000
            )
            await username_input.fill(USERNAME)
            
            # Click Next
            await asyncio.sleep(1)
            next_button = await page.wait_for_selector(
                "button[role='button']:has-text('Next')",
                timeout=5000
            )
            await next_button.click()
            
            # Wait for next step
            await asyncio.sleep(3)
            
            # Check if email verification is needed
            try:
                email_input = await page.wait_for_selector(
                    "input[autocomplete='email']",
                    timeout=5000
                )
                await email_input.fill(EMAIL)
                await asyncio.sleep(1)
                buttons = await page.query_selector_all("button[role='button']")
                for btn in buttons:
                    text = await btn.text_content()
                    if text and 'Next' in text:
                        await btn.click()
                        break
                await asyncio.sleep(3)
            except:
                pass
            
            print("Entering password...")
            # Enter password
            password_input = await page.wait_for_selector(
                "input[autocomplete='current-password']",
                timeout=10000
            )
            await password_input.fill(PASSWORD)
            
            # Click Log in
            await asyncio.sleep(1)
            login_button = await page.wait_for_selector(
                "button[role='button']:has-text('Log in')",
                timeout=5000
            )
            await login_button.click()
            
            # Wait for login to complete
            print("Waiting for login to complete...")
            await asyncio.sleep(15)
            
            # Check current URL
            current_url = page.url
            print(f"Current URL: {current_url}")
            
            # Get cookies
            cookies = await context.cookies()
            print(f"Got {len(cookies)} cookies")
            
            # Save cookies
            with open(COOKIES_FILE, 'w') as f:
                json.dump(cookies, f)
            print(f"Cookies saved to {COOKIES_FILE}")
            
            return True
            
        except Exception as e:
            print(f"Error during login: {e}")
            import traceback
            traceback.print_exc()
            
            # Take screenshot
            await page.screenshot(path="login_error.png")
            print("Screenshot saved to login_error.png")
            return False
            
        finally:
            await browser.close()


if __name__ == "__main__":
    print("=" * 50)
    print("Twitter Login with Playwright")
    print("=" * 50)
    print("A browser window will open. Please complete any CAPTCHA if required.")
    print("=" * 50)
    
    success = asyncio.run(login_and_get_cookies())
    
    if success:
        print("\n✓ Login successful! Cookies saved.")
        print("You can now use the cookies with twikit for scraping.")
    else:
        print("\n✗ Login failed. Check the error details above.")
