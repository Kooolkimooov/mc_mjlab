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
| [architecture/](architecture/README.md) | how the system fits together, **generated from the source** by `scripts/generate_architecture_docs.py` |
| [hard-constraints.md](hard-constraints.md) | numbered constraints every config change must cite, and the decision journal |
| [leo-mjlab-review.md](leo-mjlab-review.md) | exhaustive adoption matrix and implementation order from `leo_mjlab` commits `1fcdcee..579183f` |
| [improvement-roadmap.md](improvement-roadmap.md) | ordered critique remediation, acceptance gates, and experiment program |
| [improvement-screens.md](improvement-screens.md) | critique screen measurements, rejected variants, and qualification shortlist |
| [corrective-policy-review.md](corrective-policy-review.md) | branch-review disposition, deferred defects, and the active-only policy correction |
| [difficulty.md](difficulty.md) | `push_velocity`, `push_angular_velocity`, `warmup_s`, `episode_length_s` |
| [reward-shaping.md](reward-shaping.md) | reward weights and stds, live audit units and contracts, metrics, rejected ideas |
| [residual-authority.md](residual-authority.md) | residual scales, feasibility, recovery detector, authority probes |
| [ppo.md](ppo.md) | `init_std`, `std_range`, `entropy_coef`, `desired_kl`, `num_steps_per_env`, the training budget and diagnostics |
| [training-parameter-provenance.md](training-parameter-provenance.md) | root source and history of every active and pruned training parameter |
| [residual-growth.md](residual-growth.md) | `entropy_coef`, `learn_std` and `residual_magnitude` screening results |
| [observations.md](observations.md) | noise levels, the actor/critic split, the `controller_planned_*` terms |
| [external-controller-api.md](external-controller-api.md) | versioned deployment boundary for recovery state, residuals, and references |
| [walking-reference.md](walking-reference.md) | datastore velocity action, bounds, and deterministic gain screen |
| [residual-mpc.md](residual-mpc.md) | paper-style HRP5P task, action scale, objective telemetry, sign correction, and fidelity limits |
| [residual-feedback.md](residual-feedback.md) | residual on the controller's own feedback rather than its output, and why the torque channel competes with it |
| [controller-timing.md](controller-timing.md) | datastore inventory, generic scalar transport, step-duration screen, and why `controller_timeout_ms` is 60000 |
| [controller-profiling.md](controller-profiling.md) | native training CPU profile: QP preparation, kinematics, integration costs and OpenMP barriers |
| [evaluation.md](evaluation.md) | how the measurement scripts avoid biasing a result, and what each contract suite covers |
| [training-watchdog.md](training-watchdog.md) | attach protocol, escalation thresholds, qualification baseline, and machine-readable verdicts |
| [coupling.md](coupling.md) | action term, pool and host: interpolation, dispatch lag, reset ordering |
| [process-workers.md](process-workers.md) | native process backend, IPC ownership and worker recovery contract |
| [robots.md](robots.md) | collision geoms, PD gains, extra sensors, refJointOrder, assets |
| [locomanip.md](locomanip.md) | HRP5P cart demo, dependency patch, contact tuning, acceptance results |

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

`scripts/check_prose.py` enforces the other half of the arrangement — that
the code stays under 10% prose with one-line docstrings — and warns about
headings here whose identifier no longer exists.

Historical coupling measurements are retained in [coupling-history.md](coupling-history.md).
