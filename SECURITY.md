# Security Policy

## Supported Versions
This repository is currently maintained on the `main` branch.

## Reporting a Vulnerability
If you discover a security issue, please report it privately:

- Open a private security advisory in GitHub (preferred), or
- Contact the repository owner directly and include:
  - A clear description of the issue
  - Reproduction steps
  - Potential impact
  - Suggested fix (if available)

Please do **not** disclose vulnerabilities publicly until a fix is available.

## Secret & Credential Safety
This project includes protections to reduce accidental secret leakage:

- `.env.example` is a safe template and may be committed.
- `.env` and local runtime artifacts are ignored by Git.
- `results/emergency_settings.json` is ignored.
- Runtime persistence sanitizes SMTP password before writing to disk.
- `scripts/secret_guard.py` scans for common secret patterns.
- `.githooks/pre-commit` blocks commits when likely secrets are detected.

Before every push, run:

```bash
python scripts/secret_guard.py --all
```

If credentials are exposed:

1. Revoke/rotate the credential immediately.
2. Remove leaked values from code/history.
3. Force-push cleaned history only if absolutely required.
4. Document the incident and remediation.
