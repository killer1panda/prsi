"""
Use Selenium to login to Twitter/X and extract cookies.
Then use those cookies with twikit for scraping.
"""

import json
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Credentials
EMAIL = "vaasha038@gmail.com"
USERNAME = "doomlord14686"
PASSWORD = "Hesoyam1@"

COOKIES_FILE = "twitter_cookies.json"


def setup_driver():
    """Setup Chrome driver with stealth options."""
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    # Use installed Chrome
    chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # Stealth options
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # User agent
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Make selenium less detectable
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    return driver


def login_and_get_cookies():
    """Login to Twitter and get cookies."""
    driver = setup_driver()
    
    try:
        print("Opening Twitter login page...")
        driver.get("https://twitter.com/i/flow/login")
        
        # Wait for login form to load
        time.sleep(5)
        
        print("Entering username...")
        # Enter username/email
        username_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[autocomplete='username']"))
        )
        username_input.send_keys(USERNAME)
        
        # Click next
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR, "button[role='button']").click()
        
        # Wait for next step
        time.sleep(3)
        
        # Check if we need to enter email or username
        try:
            # Try to find email input if it asks for email verification
            email_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[autocomplete='email']"))
            )
            email_input.send_keys(EMAIL)
            time.sleep(1)
            driver.find_element(By.CSS_SELECTOR, "button[role='button']").click()
            time.sleep(3)
        except:
            pass
        
        # Enter password
        print("Entering password...")
        password_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[autocomplete='current-password']"))
        )
        password_input.send_keys(PASSWORD)
        
        # Click login
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR, "button[role='button']").click()
        
        # Wait for login to complete
        print("Waiting for login to complete...")
        time.sleep(10)
        
        # Get cookies
        cookies = driver.get_cookies()
        print(f"Got {len(cookies)} cookies")
        
        # Save cookies
        with open(COOKIES_FILE, 'w') as f:
            json.dump(cookies, f)
        print(f"Cookies saved to {COOKIES_FILE}")
        
        return True
        
    except Exception as e:
        print(f"Error during login: {e}")
        # Take screenshot for debugging
        driver.save_screenshot("login_error.png")
        print("Screenshot saved to login_error.png")
        return False
        
    finally:
        driver.quit()


if __name__ == "__main__":
    print("=" * 50)
    print("Twitter Login with Selenium")
    print("=" * 50)
    
    success = login_and_get_cookies()
    
    if success:
        print("\nLogin successful! Cookies saved.")
    else:
        print("\nLogin failed. Check login_error.png for details.")
