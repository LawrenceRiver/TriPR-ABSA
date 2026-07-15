# Security policy

## Reporting a vulnerability

Please do not publish API keys, credentials, private data, or an unpatched
vulnerability in a public issue. Use GitHub's private vulnerability reporting
form instead:

https://github.com/LawrenceRiver/TriPR-ABSA/security/advisories/new

Include the affected file or component, reproduction steps, and the impact you
observed. Do not include credentials or private datasets in the report.

## Credential handling

TriPR-ABSA does not require an API key for inference, tests, visualization, or
offline prior construction. The optional DeepSeek provider reads
`DEEPSEEK_API_KEY` from the process environment. The key must not be placed in a
source file, command example, prior JSON, issue, or commit.

Before publishing a change, scan both the working tree and Git history for
secrets. If a real credential is committed, revoke it first; deleting the line
in a later commit does not remove it from history.
