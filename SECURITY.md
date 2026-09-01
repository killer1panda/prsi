# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.0.x   | :white_check_mark: |
| 2.0.x   | :x:                |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take the security of the Doom Index very seriously. Due to the sensitive nature of predictive modeling, if you discover a security vulnerability, **please DO NOT open a public issue.**

Instead, report it privately by emailing: `security@doomindex.internal`

### Response Protocol
1. We will acknowledge receipt of your vulnerability report within 24 hours.
2. We will provide a triage timeline within 48 hours.
3. Upon verifying the vulnerability, we will release a patch and publish a security advisory.

### Scope
- Causal Inference Model poisoning attacks
- API Gateway (Kong) bypass vulnerabilities
- Federated Learning data leaks
- Hardcoded secrets or Infrastructure-as-Code (Terraform) exposure
