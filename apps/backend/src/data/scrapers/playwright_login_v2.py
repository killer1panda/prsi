"""
Use Playwright to login to Twitter/X and extract cookies.
Try x.com and with proxy support.
"""

import asyncio
import json
from playwright.async_api import async_playwright

# Credentials
EMAIL = "vaasha038@gmail.com"
USERNAME = "doomlord14686"
PASSWORD = "Hesoyam1@"

COOKIES_FILE = "twitter_cookies.json"

# Try with a proxy
PROXY = "http://95.217.195.146:9999"


async def login_and_get_cookies():
    """Login to Twitter and get cookies using Playwright."""
    print("Starting Playwright...")
    
    async with async_playwright() as p:
        # Launch browser 
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox'
            ]
        )
        
        # Create context with stealth settings and proxy
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            proxy={"server": PROXY}
        )
        
        # Add stealth scripts
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        
        # Enable request interception for debugging
        page.on("request", lambda request: print(f"Request: {request.url}"))
        page.on("response", lambda response: print(f"Response: {response.status} {response.url}"))
        
        try:
            print("Opening X login page...")
            # Try x.com first
            try:
                await page.goto("https://x.com/i/flow/login", timeout=60000)
            except:
                print("x.com failed, trying twitter.com...")
                await page.goto("https://twitter.com/i/flow/login", timeout=60000)
            
            # Wait a bit for page to load
            await asyncio.sleep(10)
            
            # Check if we got blocked
            if "challenge" in page.url.lower() or "blocked" in (await page.title()).lower():
                print("Got blocked by Cloudflare!")
                await page.screenshot(path="blocked.png")
                return False
            
            print("Entering username...")
            # Try to find username input
            try:
                username_input = await page.wait_for_selector(
                    "input[autocomplete='username']", 
                    timeout=15000
                )
            except:
                # Try alternative selectors
                username_input = await page.wait_for_selector(
                    "input[type='text']",
                    timeout=15000
                )
            
            await username_input.fill(USERNAME)
            
            # Click Next
            await asyncio.sleep(1)
            buttons = await page.query_selector_all("button[role='button']")
            for btn in buttons:
                text = await btn.text_content()
                if text and 'Next' in text:
                    await btn.click()
                    break
            
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
                timeout=15000
            )
            await password_input.fill(PASSWORD)
            
            # Click Log in
            await asyncio.sleep(1)
            buttons = await page.query_selector_all("button[role='button']")
            for btn in buttons:
                text = await btn.text_content()
                if text and ('Log in' in text or 'Sign in' in text):
                    await btn.click()
                    break
            
            # Wait for login to complete
            print("Waiting for login to complete...")
            await asyncio.sleep(20)
            
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
    print("Twitter Login with Playwright v2")
    print("=" * 50)
    print("A browser window will open. Please complete any CAPTCHA if required.")
    print("=" * 50)
    
    success = asyncio.run(login_and_get_cookies())
    
    if success:
        print("\n✓ Login successful! Cookies saved.")
        print("You can now use the cookies with twikit for scraping.")
    else:
        print("\n✗ Login failed. Check the error details above.")
