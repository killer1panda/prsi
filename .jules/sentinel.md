## 2025-03-01 - Hardcoded Scraper Credentials Removed
**Vulnerability:** Found multiple hardcoded credentials (email, username, password) used by Playwright, Selenium, and un-detected chromedriver scripts for Twitter scraping.
**Learning:** Social scraping scripts in data extraction folders often embed dummy or real credentials for rapid iterations, which leaks credentials and exposes secrets in the source code.
**Prevention:** Avoid embedding hardcoded plain-text developer credentials. Use `os.getenv` passing the variables (e.g. `TWITTER_EMAIL`, `TWITTER_USERNAME`, `TWITTER_PASSWORD`) and rely on environment variables instead.
