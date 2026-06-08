# Schema Versioning Rules

## Version format

All contract schemas include a `_contract_version` field using [semantic versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

## When to bump

| Change type | Version bump | Example |
|---|---|---|
| Remove a field | **MAJOR** | Dropping `metar_wind_kt` |
| Rename a field | **MAJOR** | `baro_altitude_m` → `altitude_m` |
| Tighten a constraint (narrower range, stricter pattern) | **MAJOR** | `lat` max from 90 → 85 |
| Add a new required field | **MAJOR** | Adding `geo_altitude_m` as required |
| Add a new optional field | **MINOR** | Adding `spi` (boolean, nullable) |
| Loosen a constraint (wider range) | **MINOR** | `velocity_ms` min from 0 → -5 |
| Fix a typo in description | **PATCH** | Correcting a comment |
| Update `_contract_notes` | **PATCH** | Clarifying semantics |

## Rules

1. **The `silver_flight_state` schema is the shared spine.** All four layers depend on it. Breaking changes require owner approval and a coordinated migration across layers.

2. **Gold schemas are downstream contracts.** They may evolve faster but must remain compatible with the silver schema they derive from.

3. **`additionalProperties: false` is enforced.** New fields must be added to the schema before they can appear in data. This prevents silent schema drift.

4. **Fixture files must be updated alongside schema changes.** Every schema change must include updated valid/invalid fixtures that exercise the change.

5. **`make test-contracts` must pass before any schema change is committed.** This is enforced in CI.

6. **Version history is tracked in git.** There is no separate changelog — use `git log --oneline shared/contracts/` to see the evolution.
