#!/usr/bin/env python3
"""
Get fresh Twitter cookies using undetected-chromedriver
This should bypass bot detection.
"""

import time
import json
from pathlib import Path
from datetime import datetime

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
TWITTER_USERNAME = "doomlord14686"
TWITTER_EMAIL = "vaasha038@gmail.com"  
TWITTER_PASSWORD = "Hesoyam1@"

OUTPUT_DIR = Path('doom-index/doom-index/data/twitter_dataset')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def login_and_get_cookies():
    """Login to Twitter and extract cookies."""
    print("[UC] Starting browser...")
    
    driver = uc.Chrome(headless=False, use_subprocess=True)
    
    try:
        print("[UC] Opening Twitter login...")
        driver.get('https://twitter.com/i/flow/login')
        
        # Wait for page to load
        time.sleep(5)
        
        # Enter username
        print("[UC] Entering username...")
        try:
            username_input = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
            )
        except:
            username_input = driver.find_element(By.CSS_SELECTOR, 'input[name="text"]')
        
        username_input.click()
        time.sleep(1)
        username_input.send_keys(TWITTER_USERNAME)
        
        # Click Next
        time.sleep(0.5)
        try:
            next_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Next')]")
            next_btn.click()
        except:
            driver.find_element(By.TAG_NAME, 'body').send_keys('\n')
        
        time.sleep(3)
        
        # Check if email is needed
        try:
            print("[UC] Entering email...")
            email_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"]'))
            )
            email_input.send_keys(TWITTER_EMAIL)
            time.sleep(1)
            
            try:
                next_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Next')]")
                next_btn.click()
            except:
                driver.find_element(By.TAG_NAME, 'body').send_keys('\n')
            
            time.sleep(3)
        except Exception as e:
            print(f"[UC] Email step skipped: {e}")
        
        # Enter password
        print("[UC] Entering password...")
        try:
            password_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="password"]'))
            )
        except:
            password_input = driver.find_element(By.CSS_SELECTOR, 'input[name="password"]')
        
        password_input.send_keys(TWITTER_PASSWORD)
        
        time.sleep(1)
        
        # Click Login
        try:
            login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Log in')]")
            login_btn.click()
        except:
            driver.find_element(By.TAG_NAME, 'body').send_keys('\n')
        
        print("[UC] Waiting for login...")
        time.sleep(8)
        
        # Check if we're logged in
        current_url = driver.current_url
        print(f"[UC] Current URL: {current_url}")
        
        # Get all cookies
        print("[UC] Extracting cookies...")
        cookies = driver.get_cookies()
        
        # Save cookies
        cookie_file = OUTPUT_DIR / 'uc_cookies.json'
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        print(f"[UC] Saved {len(cookies)} cookies to {cookie_file}")
        
        # Print key cookies
        cookies_dict = {c['name']: c['value'] for c in cookies}
        print("\n[UC] Key cookies:")
        for name in ['auth_token', 'ct0', 'session', 'guest_id']:
            if name in cookies_dict:
                print(f"  - {name}: {cookies_dict[name][:20]}...")
        
        return cookies_dict
        
    finally:
        input("Press Enter to close browser...")
        driver.quit()


if __name__ == '__main__':
    print("=" * 60)
    print("UNDETECTED CHROMEDRIVER AUTHENTICATOR")
    print("=" * 60)
    print("\nThis will open a browser window.")
    print("Please log in manually if automation fails.")
    print()
    
    try:
        cookies = login_and_get_cookies()
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Total cookies: {len(cookies)}")
        print(f"\nSaved to: {OUTPUT_DIR / 'uc_cookies.json'}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
