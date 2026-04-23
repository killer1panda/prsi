#!/usr/bin/env python3
"""
Get fresh Twitter cookies using Selenium
This opens a browser, logs in, and extracts cookies.
"""

import time
import json
from pathlib import Path
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("Installing selenium...")
    import subprocess
    subprocess.run(['pip3', 'install', 'selenium'], check=True)
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

# Configuration
TWITTER_USERNAME = "doomlord14686"
TWITTER_EMAIL = "vaasha038@gmail.com"  
TWITTER_PASSWORD = "Hesoyam1@"

OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_chrome_driver():
    """Create Chrome driver with stealth settings."""
    options = Options()
    
    # Stealth options
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Common options
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1280,720')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    
    # Stealth script
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    return driver


def login_and_get_cookies():
    """Login to Twitter and extract cookies."""
    print("[SELENIUM] Starting browser...")
    driver = get_chrome_driver()
    
    try:
        print("[SELENIUM] Opening Twitter...")
        driver.get('https://twitter.com/i/flow/login')
        
        # Wait for page to load
        time.sleep(3)
        
        # Enter username
        print("[SELENIUM] Entering username...")
        username_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_input.click()
        time.sleep(0.5)
        username_input.send_keys(TWITTER_USERNAME)
        
        # Click Next
        time.sleep(0.5)
        try:
            next_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Next')]")
            next_btn.click()
        except:
            driver.find_element(By.TAG_NAME, 'body').send_keys('\n')
        
        time.sleep(2)
        
        # Check if email is needed
        try:
            print("[SELENIUM] Entering email...")
            email_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"]'))
            )
            email_input.send_keys(TWITTER_EMAIL)
            time.sleep(0.5)
            
            try:
                next_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Next')]")
                next_btn.click()
            except:
                driver.find_element(By.TAG_NAME, 'body').send_keys('\n')
            
            time.sleep(2)
        except Exception as e:
            print(f"[SELENIUM] Email step skipped: {e}")
        
        # Enter password
        print("[SELENIUM] Entering password...")
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="password"]'))
        )
        password_input.send_keys(TWITTER_PASSWORD)
        
        time.sleep(0.5)
        
        # Click Login
        try:
            login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Log in')]")
            login_btn.click()
        except:
            driver.find_element(By.TAG_NAME, 'body').send_keys('\n')
        
        print("[SELENIUM] Waiting for login...")
        time.sleep(5)
        
        # Check if we're logged in
        current_url = driver.current_url
        print(f"[SELENIUM] Current URL: {current_url}")
        
        if 'home' in current_url or 'i/flow/login' not in current_url:
            print("[SELENIUM] ✓ Logged in!")
        else:
            print("[SELENIUM] ⚠️ May not be fully logged in")
        
        # Get all cookies
        print("[SELENIUM] Extracting cookies...")
        cookies = driver.get_cookies()
        
        # Save cookies
        cookies_dict = {c['name']: c['value'] for c in cookies}
        
        cookie_file = OUTPUT_DIR / 'selenium_cookies.json'
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        print(f"[SELENIUM] Saved {len(cookies)} cookies to {cookie_file}")
        
        # Print key cookies
        print("\n[SELENIUM] Key cookies:")
        for name in ['auth_token', 'ct0', 'session', 'guest_id']:
            if name in cookies_dict:
                print(f"  - {name}: {cookies_dict[name][:20]}...")
        
        return cookies_dict
        
    finally:
        driver.quit()


def get_bearer_from_page():
    """Try to get bearer token from browser network."""
    print("\n[SELENIUM] Attempting to get bearer token...")
    driver = get_chrome_driver()
    
    try:
        driver.get('https://twitter.com/explore')
        time.sleep(3)
        
        # Get all cookies
        cookies = driver.get_cookies()
        
        # Try to find bearer in local storage
        driver.get('https://twitter.com/i/flow/login')
        time.sleep(2)
        
        return {c['name']: c['value'] for c in cookies}
        
    finally:
        driver.quit()


if __name__ == '__main__':
    print("=" * 60)
    print("SELENIUM AUTHENTICATOR")
    print("=" * 60)
    
    try:
        cookies = login_and_get_cookies()
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Total cookies: {len(cookies)}")
        print(f"\nSaved to: {OUTPUT_DIR / 'selenium_cookies.json'}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
