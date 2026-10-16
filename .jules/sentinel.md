## 2026-09-04 - Timing Attack Vulnerability in API Key Verification
**Vulnerability:** The API key validation in `verify_api_key` (`api_v2_production.py`) compared the provided credentials against valid keys using the `in` operator (list membership check), which uses standard string comparison. Standard string comparison short-circuits on the first differing character, creating a timing side-channel that could allow an attacker to guess the API key character-by-character.
**Learning:** Even simple string comparisons for secrets (like `api_key in valid_keys` or `api_key == valid_key`) can expose the application to timing attacks. This is a common pattern when developers manually validate secrets without using established cryptographic functions.
**Prevention:** Always use `secrets.compare_digest(a, b)` for comparing sensitive strings like passwords, API keys, or tokens. This function compares strings in constant time, eliminating the timing side-channel.

## 2025-03-09 - Hardcoded Twitter Credentials in Multiple Scrapers
**Vulnerability:** Found multiple scraper scripts (selenium_login.py, run_twitter_scraper.py, playwright_login.py, playwright_login_v2.py, uc_login.py) that contained hardcoded plain-text developer credentials for Twitter (email, username, password) which would be exposed in the repository.
**Learning:** Hardcoded credentials are often copied across multiple similar scripts as different approaches are attempted (e.g., trying different scraping tools like Selenium, Playwright, or Twikit). Finding one hardcoded secret likely means others exist in sibling scripts.
**Prevention:** All scripts, even one-off helper or test scripts, should retrieve credentials using environment variables (`os.environ.get()` or a centralized config manager) instead of hardcoding them. Establish a pre-commit hook to scan for sensitive tokens.

## 2024-05-18 - API Key Authentication Bypass via Empty String Splitting
**Vulnerability:** Empty API_KEYS environment variable can lead to an authentication bypass. Python's `"".split(",")` evaluates to `['']`. If a client sends an empty Bearer token, `secrets.compare_digest("", "")` evaluates to `True`, successfully authenticating the request.
**Learning:** When parsing configuration lists from environment variables, `split()` can return a list containing an empty string rather than an empty list, which can unexpectedly match empty user input when using strict equality or digest comparisons.
**Prevention:** Always filter out falsy values (e.g. `if not key: continue` or list comprehensions) when splitting strings into lists of credentials to prevent empty credentials from successfully matching empty configuration states.
