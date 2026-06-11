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
| `v0.2.2` | Codebase language normalization | Engineering Task 2.5.2 is complete; developer-facing comments, docstrings, logs, exceptions, and tests now use consistent English. |
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

For this project, Story 2.1 through Story 2.5 together form a good `v0.2.0` milestone because they complete the RAG runtime foundation. Story 2.5.1 and Engineering Task 2.5.2 are patch-level quality milestones that harden and polish the runtime before moving into agent orchestration.

## Public Planning Policy

Public documentation should show enough planning for collaborators and reviewers to understand direction without publishing every internal backlog detail too early.

- `docs/stories/` contains completed stories and the next near-term story or engineering task.
- `docs/roadmap.md` contains the broader Epic 3, Epic 4, and Epic 5 direction.
- Future detailed story specs are added when the team is ready to implement or invite contribution on that work.
- GitHub Issues can be used later to expose contributor-friendly tasks without turning every internal planning item into a permanent public spec.

## Future Epic Direction

### Epic 3: Agent Workflow Prototype

Goal: Build the first multi-agent reasoning workflow on top of the hardened RAG runtime.

Planned direction:

- LangGraph state graph foundation.
- Researcher agent that can query retrieved context.
- Reporter agent that can produce grounded answers.
- Web search tool integration through MCP when appropriate.
- Reviewer or quality gate agent for answer validation.

### Epic 4: Streaming User Experience

Goal: Make the system usable as an interactive demo with transparent reasoning and citation-aware answers.

Planned direction:

- Streamlit visual theme and application shell.
- FastAPI server-sent events for streaming progress.
- Reasoning trace UI for agent steps.
- Markdown answer rendering with citations.
- User-friendly error and fallback messaging for recoverable runtime failures.

### Epic 5: Deployable Demo

Goal: Prepare the project for interview review and hosted demonstration.

Planned direction:

- Dockerized runtime suitable for Hugging Face Spaces or similar hosting.
- GitHub Actions quality checks.
- Release preparation, documentation cleanup, and demo readiness.
- Versioned GitHub Releases for meaningful milestones.

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
| 2.5.1 | Runtime hardening and harness guardrails | Complete |
| 2.5.2 | Engineering task: codebase language normalization | Complete |

## Known Scope Gaps & Backlog

- [ ] [Issue #1](https://github.com/AlgoMonokuma/multi-agent-rag-assistant/issues/1): PDF image and table support (scope gap identified after Story 2.1, targeting v0.2.0)
