## 2024-05-18 - Avoid Hardcoding Neo4j Credentials
**Vulnerability:** Hardcoded database credentials found in code such as default password values (e.g. `"password"`).
**Learning:** Default values used when an environment variable is missing often result in exposed secrets when instances run locally.
**Prevention:** Rely entirely on environment variables, using fallback logic that fails explicitly if missing (or using a secure secret manager).
