# Security policy

## Supported versions

Security fixes target the latest public release and default branch.

## Reporting a vulnerability

Do not open a public issue containing credentials, private data, exploit
details, or a vulnerable endpoint. Use the repository's private vulnerability
reporting or Security Advisory flow. If unavailable, contact the repository
owner privately for a secure reporting channel.

Include:

- the affected commit and file;
- the smallest safe reproduction;
- impact and required preconditions;
- whether credentials, personal data, model remote code, or a network service
  are involved;
- any proposed mitigation.

Do not test credentials or third-party services without authorization.

## Operational security

- Do not commit `.env`, access tokens, endpoint credentials, datasets, run
  artifacts, model weights, or checkpoints.
- Model loaders that execute Hub remote code require an immutable reviewed
  revision. Treat model and dataset downloads as supply-chain inputs.
- Hosted scoring sends query/document text to the configured endpoint. Use only
  approved HTTPS endpoints and data.
- Scan final container images and dependency sets before distribution or
  deployment.

## Disclosure

The maintainers will assess complete private reports, coordinate fixes, and
agree on disclosure timing with the reporter where possible. No response-time
service level is promised.
