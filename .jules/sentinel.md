## 2026-09-04 - Timing Attack Vulnerability in API Key Verification
**Vulnerability:** The API key validation in `verify_api_key` (`api_v2_production.py`) compared the provided credentials against valid keys using the `in` operator (list membership check), which uses standard string comparison. Standard string comparison short-circuits on the first differing character, creating a timing side-channel that could allow an attacker to guess the API key character-by-character.
**Learning:** Even simple string comparisons for secrets (like `api_key in valid_keys` or `api_key == valid_key`) can expose the application to timing attacks. This is a common pattern when developers manually validate secrets without using established cryptographic functions.
**Prevention:** Always use `secrets.compare_digest(a, b)` for comparing sensitive strings like passwords, API keys, or tokens. This function compares strings in constant time, eliminating the timing side-channel.

## 2025-03-09 - Hardcoded Twitter Credentials in Multiple Scrapers
**Vulnerability:** Found multiple scraper scripts (selenium_login.py, run_twitter_scraper.py, playwright_login.py, playwright_login_v2.py, uc_login.py) that contained hardcoded plain-text developer credentials for Twitter (email, username, password) which would be exposed in the repository.
**Learning:** Hardcoded credentials are often copied across multiple similar scripts as different approaches are attempted (e.g., trying different scraping tools like Selenium, Playwright, or Twikit). Finding one hardcoded secret likely means others exist in sibling scripts.
**Prevention:** All scripts, even one-off helper or test scripts, should retrieve credentials using environment variables (`os.environ.get()` or a centralized config manager) instead of hardcoding them. Establish a pre-commit hook to scan for sensitive tokens.

## 2025-03-09 - Authentication Bypass via Empty Environment Variable Split
**Vulnerability:** The API key validation used `os.environ.get("API_KEYS", "").split(",")` which evaluates to `['']` when empty. This allowed an attacker to bypass authentication by sending an empty string as the credentials, as `secrets.compare_digest("", "")` returns `True`.
**Learning:** Using string splitting on empty strings yields a list with an empty string, not an empty list. When this is used in credential validation, it can unintentionally create an authentication bypass.
**Prevention:** Always filter out empty strings when parsing environment variables for security tokens (e.g., `[k for k in env_var.split(",") if k]`). Also ensure that provided credentials are not empty before comparing them.
