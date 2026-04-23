"""
Use undetected-chromedriver to login to Twitter/X and extract cookies.
"""

import json
import time
import os
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Credentials
EMAIL = "vaasha038@gmail.com"
USERNAME = "doomlord14686"
PASSWORD = "Hesoyam1@"

COOKIES_FILE = "twitter_cookies.json"


def login_and_get_cookies():
    """Login to Twitter and get cookies using undetected-chromedriver."""
    print("Setting up undetected-chromedriver...")
    
    driver = uc.Chrome(headless=False, version_main=None)
    
    try:
        print("Opening Twitter login page...")
        driver.get("https://twitter.com/i/flow/login")
        
        # Wait for login form to load
        time.sleep(8)
        
        print("Entering username...")
        # Enter username
        username_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[autocomplete='username']"))
        )
        username_input.send_keys(USERNAME)
        
        # Click next
        time.sleep(1)
        buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='button']")
        for btn in buttons:
            if btn.text and 'Next' in btn.text:
                btn.click()
                break
        else:
            # Just click the first button if we can't find "Next"
            buttons[0].click()
        
        # Wait for next step
        time.sleep(3)
        
        # Check if we need to enter email for verification
        try:
            email_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[autocomplete='email']"))
            )
            email_input.send_keys(EMAIL)
            time.sleep(1)
            buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='button']")
            for btn in buttons:
                if btn.text:
                    btn.click()
                    break
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
        buttons = driver.find_elements(By.CSS_SELECTOR, "button[role='button']")
        for btn in buttons:
            if btn.text and ('Log in' in btn.text or 'Sign in' in btn.text):
                btn.click()
                break
        
        # Wait for login to complete
        print("Waiting for login to complete...")
        time.sleep(15)
        
        # Check if we're logged in
        current_url = driver.current_url
        print(f"Current URL: {current_url}")
        
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
        import traceback
        traceback.print_exc()
        
        # Take screenshot for debugging
        try:
            driver.save_screenshot("login_error.png")
            print("Screenshot saved to login_error.png")
        except:
            pass
        return False
        
    finally:
        driver.quit()


if __name__ == "__main__":
    print("=" * 50)
    print("Twitter Login with Undetected ChromeDriver")
    print("=" * 50)
    print("A browser window will open. Please complete any CAPTCHA if required.")
    print("=" * 50)
    
    success = login_and_get_cookies()
    
    if success:
        print("\n✓ Login successful! Cookies saved.")
        print("You can now use the cookies with twikit for scraping.")
    else:
        print("\n✗ Login failed. Check the error details above.")
