# Security policy

Please report vulnerabilities privately through GitHub's security advisory
feature rather than a public issue.

The Scholar-facing collector never receives the Hugging Face token. The upload
workflow checks out trusted publisher code from `main`, validates a checksummed
snapshot from the automation branch, and only then enters the protected
`huggingface` environment.

Do not commit API tokens, proxy credentials, cookies, or CAPTCHA-solving
services. This project intentionally does not support bypassing Google Scholar
access controls.
