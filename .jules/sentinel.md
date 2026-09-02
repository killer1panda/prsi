## 2025-03-02 - Twitter Scraper Hardcoded Credentials Leak
**Vulnerability:** Found multiple scraper scripts (e.g. `playwright_login.py`, `selenium_login.py`) leaking an explicit personal Twitter email, username, and password in plain text.
**Learning:** Development-time scaffolding logic for logging into external systems was merged directly into the codebase without transitioning credentials to a secure secrets manager or `.env` configuration.
**Prevention:** Establish pre-commit hooks to scan for credential signatures (e.g., matching standard parameter names like `PASSWORD = "..."`) prior to allowing merges on scraper utilities.
