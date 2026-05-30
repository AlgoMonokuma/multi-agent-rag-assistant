# Roadmap and Release Strategy

## Current Recommendation

Do not wait until the whole project is finished before publishing documentation. For an interview-facing side project, it is better to push clean documentation as the project evolves because it shows iteration, planning, and engineering discipline.

The right pattern is:

```text
story -> focused commit -> tests -> changelog update -> push
milestone -> tag -> GitHub Release
```

Stories should be pushed as regular commits. GitHub Releases should be reserved for meaningful milestones.

## Versioning Plan

| Version | Theme | Release When |
| --- | --- | --- |
| `v0.1.0` | Project foundation | Setup, config, logging, initial API/UI bootstrap are stable. |
| `v0.2.0` | RAG runtime foundation | Parser, indexing, chunking, hybrid retrieval, profiles, and re-ranking are documented and tested. |
| `v0.2.1` | RAG runtime hardening | Story 2.5.1 guardrails are complete and tests pass. |
| `v0.3.0` | Agent workflow prototype | LangGraph state graph and first agent workflow are usable. |
| `v0.4.0` | Streaming user experience | FastAPI SSE and Streamlit trace UI are usable. |
| `v0.5.0` | Deployable demo | Docker, CI, and hosting path are ready. |
| `v1.0.0` | Interview demo release | End-to-end demo is stable, documented, and easy to run. |

## How Many Stories Per Release?

Use commits for every completed story. Use releases only when a reviewer can understand a meaningful capability.

Recommended cadence:

- Push every story or small group of related fixes.
- Create a release after 3 to 6 related stories, or after one complete project capability.
- Create patch releases for hardening, bug fixes, documentation cleanup, or test improvements.

For this project, Story 2.1 through Story 2.5 together form a good `v0.2.0` milestone because they complete the RAG runtime foundation.

## GitHub Release Checklist

Before creating a release:

1. Confirm README and docs are current.
2. Update `CHANGELOG.md`.
3. Run relevant tests.
4. Commit all release-related documentation and code.
5. Create a version tag.
6. Push the tag.
7. Draft GitHub Release notes from `CHANGELOG.md`.

Example commands:

```powershell
git tag v0.2.0
git push origin v0.2.0
```

If the GitHub CLI is available, a release can be created with:

```powershell
gh release create v0.2.0 --title "v0.2.0 - RAG Runtime Foundation" --notes-file CHANGELOG.md
```

## Story Status

| Story | Title | Public Status |
| --- | --- | --- |
| 1.1 | Project foundation initialization | Complete |
| 1.2 | TDD test environment | Complete |
| 1.3 | Secure config loading | Complete |
| 1.4 | Shared logging and engineering docs | Complete |
| 2.1 | Document ingestion and parser pipeline | Complete |
| 2.2 | Session-isolated indexing foundation | Complete |
| 2.3 | Text chunking and embedding pipeline | Complete |
| 2.4 | Hybrid search implementation | Complete |
| 2.4.5 | Document-type chunking profile | Complete |
| 2.5 | Re-ranking mechanism | Complete |
| 2.5.1 | Runtime hardening and harness guardrails | Planned |
