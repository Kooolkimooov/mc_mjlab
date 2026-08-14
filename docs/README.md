This directory is mostly here for agents to documents findings and choices
out of source code in order to keep it readable.

# Why the numbers are what they are

The measurement record behind this repo's constants and design choices. It lives
here rather than in comments because it is a lab notebook: it matters when you
are *deciding* something, not when you are reading the line that implements it.

| Where | Holds |
| --- | --- |
| `README.md` | how to use the repo |
| `CLAUDE.md` | how to work in it; hazards you need before touching anything |
| `docs/` (here) | why a specific number or design is what it is |

## Index

| File | Covers |
| --- | --- |
| [difficulty.md](difficulty.md) | `PUSH_VELOCITY`, `PUSH_ANGULAR_VELOCITY`, `PUSH_WARMUP_S`, `WALK_WINDOW_S` |
| [reward-shaping.md](reward-shaping.md) | tracking weights and stds, the termination penalty, metrics, rejected ideas |
| [residual-authority.md](residual-authority.md) | `residual_scale`, the per-joint partition, the torque budget |
| [ppo.md](ppo.md) | `init_std`, `std_range`, `entropy_coef`, `desired_kl`, `num_steps_per_env` |
| [observations.md](observations.md) | noise levels, the actor/critic split, the `controller_planned_*` terms |
| [evaluation.md](evaluation.md) | how the measurement scripts avoid biasing a result |
| [coupling.md](coupling.md) | action term, pool and host: interpolation, dispatch lag, reset ordering |
| [robots.md](robots.md) | collision geoms, PD gains, extra sensors, refJointOrder, assets |

## Finding a note

Headings **are** the identifier, so the fastest lookup is a grep:

```sh
grep -rn PUSH_VELOCITY docs/
```

The code also carries a link where a note exists, sharing the one comment line
with the terse reason:

```python
# Difficulty dial; the baseline should almost always fail. docs/difficulty.md#push-velocity
PUSH_VELOCITY = 0.4
```

Grep is the primary route and the link is the convenience, deliberately: a link
can rot, a heading that is the identifier cannot go missing without the note
itself going missing.

Which is why the link is **skipped where the heading is already the function's
own name** — `grep` finds `zmp_tracking` from the code either way, and the
comment budget is better spent where the connection is not guessable.

## Writing a note

One `##` section per identifier, three fixed fields, so an update appends a
bullet instead of rewriting a paragraph:

```markdown
## SOME_CONSTANT

**Current:** `0.4` — one sentence on what it buys.

**Re-measure if:** what would invalidate the number.

**History:**
- 2026-07-31 — what was run, what it showed.
```

Keep the measurements verbatim when moving them. The numbers are the asset; a
paraphrase that drops the sample size is worth much less than the original.

`.claude/hooks/check_prose.py` enforces the other half of the arrangement — that
the code stays under 10% prose with one-line docstrings — and warns about
headings here whose identifier no longer exists.
