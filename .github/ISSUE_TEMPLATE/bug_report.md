---
name: Bug report
about: Report broken behaviour in the AI Document Analyst
title: "[BUG] "
labels: bug
assignees: ""
---

## What happened

<!-- A one-paragraph summary. The maintainer should be able to
     reproduce the problem from this paragraph alone. -->

## Reproduction steps

1.
2.
3.

## Expected behaviour

<!-- What you thought would happen. -->

## Actual behaviour

<!-- What actually happened. Include the full traceback
     (Stack trace, exception type, message). -->

```
<paste traceback here>
```

## Environment

| | |
|---|---|
| OS | <!-- Windows 11 / macOS 14 / Ubuntu 22.04, etc. --> |
| Python | <!-- output of `python --version` --> |
| `pip freeze` | <!-- paste the output, or at least the lines for: streamlit, httpx, pandas, numpy, matplotlib, seaborn, PyPDF2, python-docx, Pillow, pytesseract, requests, python-dotenv --> |
| App version | <!-- git describe --tags, or commit hash from `git rev-parse HEAD` --> |
| Model | <!-- the model id from Settings → Model picker, e.g. minimax-m3-free --> |
| File type | <!-- .pdf / .csv / .xlsx / .docx / .png / .txt --> |

## How to verify

<!-- Does the test suite still pass?
     Run `python -m unittest discover -s tests -v` and paste
     the summary line (e.g. "Ran 73 tests in 0.024s OK"). -->

## Additional context

<!-- Screenshots, log snippets, related issues. -->
