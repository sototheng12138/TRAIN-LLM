# Manuscript ↔ repository alignment

Use this checklist when syncing the **published paper** (Elsevier CAS single-column or final Word/LaTeX) with repository text.

| Manuscript section | Repository location |
|--------------------|----------------------|
| Title | `README.md` top-level heading |
| Highlights | `README.md` → Highlights |
| Graphical abstract | `README.md` Mermaid block; optional replacement with a static figure under `figures/` |
| Abstract | `README.md` → Abstract |
| Keywords | `README.md` → Keywords |
| Introduction | `README.md` §1 |
| Methods / setup | `README.md` §2 and `docs/Baseline_Experimental_Setup.md` |
| Results figure numbering | `README.md` §6 — replace generic labels with final figure numbers |
| Data availability | `README.md` §7 and `dataset/README_DATA.md` |
| Author contributions | Optional `AUTHORS.md` or README subsection |
| Associated-paper BibTeX | `README.md` §1 second citation block |

The PDF manuscript is the **source of truth** for wording; update the Markdown files to match after acceptance or before tagging a release.
