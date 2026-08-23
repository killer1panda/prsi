## 2026-05-18 - Hardcoded Database Credentials in Production Code

**Vulnerability:** A hardcoded password (`doom_index_prod_2026`) for the production Neo4j database was embedded directly in the `src/data/populate_neo4j_production.py` script as a default value in the `Neo4jConfig` dataclass and CLI arguments. It was also hardcoded in the `.github/workflows/ci.yml` and test files.

**Learning:** Developers sometimes use default fallback credentials in data pipeline scripts for convenience during testing, without realizing these scripts are deployed to production and expose critical infrastructure secrets in version control. The CI/CD pipeline also lacked secret management for the database integration tests.

**Prevention:** Never use hardcoded secrets as fallback values in configuration dataclasses, even for testing. Always require secrets to be injected via environment variables or secret managers, and explicitly fail if they are absent. Utilize GitHub Secrets for CI pipelines instead of plain text environment variables.
