# Story 1.3: Environment Variable and Secure Config Loading

Status: Complete

## User Story

As a developer, I want environment-driven configuration so that secrets and runtime settings are not hard-coded.

## Scope

This story introduces typed application settings and a safe example environment file.

## Acceptance Criteria

1. Given environment variables, when settings load, then values are available through a typed settings object.
2. Given missing optional values, then safe defaults are used.
3. Given local secrets, then they are excluded from git.
4. Given a new developer, then `.env.example` documents expected local configuration.

## Implementation Notes

- Settings live in `core/config.py`.
- Real local values belong in `.env`.
- `.env.example` should contain names and placeholders only.
- Tests should avoid requiring real API keys.

## Out of Scope

- Secret manager integration
- Production deployment secrets
- User authentication

## Definition of Done

- Settings can be loaded locally.
- `.env.example` is safe to commit.
- Real `.env` files are ignored.
- Config tests pass.
