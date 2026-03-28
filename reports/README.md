# Reports

All proposal and report LaTeX source files for the CS505 final project.
Build artifacts (`.aux`, `.bbl`, `.log`, etc.) are git-ignored.

---
## LaTeX installation

### macOS — MacTeX

```bash
brew install --cask mactex-no-gui
```

Or download the installer from <https://www.tug.org/mactex/>.

### Linux — TeX Live

```bash
sudo apt-get install texlive-latex-recommended texlive-fonts-recommended \
     texlive-bibtex-extra
```

### Verify

```bash
pdflatex --version && bibtex --version
```

All packages used (`times`, `microtype`, `booktabs`, `amsmath`, `amssymb`,
`natbib`, `parskip`, `hyperref`, `url`, `geometry`) ship with TeX Live 2020+
and MacTeX. No custom `.sty` files needed.

---

## Building a PDF

Run from inside `reports/`:

```bash
pdflatex proposal.tex && bibtex proposal && \
pdflatex proposal.tex && pdflatex proposal.tex
```
---

## Upcoming reports

| Milestone | Due | Pages | Format |
|---|---|---|---|
| Proposal | Mar 31, 2026 | 1–2 | LaTeX (single-column) |
| Midway Report | Apr 21, 2026 | 2–4 | ACL format |
| Final Report | May 7, 2026 | 4–6 | ACL format |

To switch to ACL format for the midway/final report, download the ACL 2023
style files from <https://github.com/acl-org/acl-style-files> into this
directory and change the `\documentclass` line accordingly.
