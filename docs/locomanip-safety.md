# Locomanip feedback safety

## SAFETY_CONFIG

**Current:** `etc/mc_rtc_hrp5_locomanip_safety.yaml` declares HRP5P and
LocomanipController, the 2 ms control period, and disabled GUI and native
logging. The selected hand adaptation (`alpha: 0.03`,
`forceProjection: [0, 0, 0.98]`, `momentProjection: [0, 0, 0]`) already lives in
the installed per-robot controller configuration. The run provenance snapshots
the selector and installed controller files. The authority gate hashes both
the selector and the installed/user controller inputs, so a changed native
tuning requires a new probe. The old 666-line project snapshot and its
authority report are
historical artifacts under `logs/locomanip/safety-2026-10-13/`; the report's
profile hash cannot admit another training run. A new authority probe is
required with the compact selector. The earlier 10/100 kg smoke check used
the removed snapshot, so it does not validate the current selector.

**Re-measure if:** the controller, robot assets, native workspace files,
user mc_rtc configuration, sensors or simulator changes.

**History:** `scripts/tune_locomanip_safety.py` measured zero-action first
episodes at seed 42 with encoder bias off, three repetitions per mass and
51.66 s episodes. The raw policy-step traces and results are under
`logs/locomanip/safety-2026-10-13/tuning/`. Each cell below is safe
completions / 3; every non-completion in this sweep was a combined fall,
collapse or controller failure, with no worker failure.

| tuning | 10 | 100 | 500 | 525 | 550 | 575 | 600 | 625 | 650 kg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03, [0,0,0.98] | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 0.05, [1,1,1] | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.06, [1,1,1] | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.07, [1,1,1] | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The full-projection settings fail the required 100 kg light-cart check, so
the current tuning wins. Under it, 525 kg completed at 49.26–49.28 s with
minimum root z 0.727–0.728 m, peak tilt 0.23–0.24 rad and maximum forward
lag 0.027–0.030 m. At 550 kg all three runs failed at 35.28–39.04 s,
with maximum lag 0.364–0.523 m. At 575 kg they failed at 28.34–29.26 s,
with maximum lag 0.357–0.543 m. The 550 kg causes were one fall, one
collapse and one controller failure; 575 kg had one fall and two collapses,
one of them also reporting controller failure. Cause labels remain separate,
while the combined outcome counts each episode once.

## native_channels

**Current:** A separate single-environment run at each of 525, 550 and
575 kg wrote one native binary controller log per episode, avoiding log
filename collisions. Each run used seed 42, encoder bias off and the
selected adaptation with native logging enabled. The logs and aligned
policy-step traces are under `logs/locomanip/safety-2026-10-13/native-*/`;
`native-summary.json` checks native end time against the episode's terminal
time. Forces below are per-hand means over the last 5 s of Hold, in each
hand's local XYZ frame (N), measured first and commanded second:

| mass | left measured | left commanded | right measured | right commanded |
| --- | --- | --- | --- | --- |
| 525 kg | (+2.7, -4.8, +20.0) | (0, 0, +11.7) | (-1.1, +2.0, +41.9) | (0, 0, +25.4) |
| 550 kg | (+9.5, -52.4, -92.0) | (0, 0, -54.2) | (+8.0, +52.3, -89.6) | (0, 0, -53.4) |
| 575 kg | (+13.1, -52.6, -111.0) | (0, 0, -66.1) | (+5.9, +49.1, -117.0) | (0, 0, -67.8) |

At the failing loads the controller commands no local X/Y force under the
selected projection even as measured lateral forces reach about 50 N per
hand. These are diagnostics of what it sent, not evidence that a new force
policy can improve safety. The single 575 kg logged run fell by tilt at
30.10 s with minimum root height 0.706 m, illustrating why height alone
cannot label failure; the three-repeat tuning result above is the selection
evidence.

**Re-measure if:** logger channel semantics, the hand frames or the chosen
adaptation changes.

## TRANSITION_MIDPOINT_KG

**Current:** The last consistently passing grid mass is 525 kg and the
first failing one is 550 kg, so `T = 537.5 kg`. Training draws 60% of
payloads log-uniformly from 462.5–612.5 kg, 20% from 10–462.5 kg and
20% from 612.5–1000 kg. `transition_payload` partitions reset IDs and
calls `dr.pseudo_inertia` once per nonempty band, retaining the body mass,
COM and inertia field declarations and `set_const` recomputation.
An empty outer band at either 10 or 1000 kg is removed and the remaining
mixture weights are renormalized.

**Re-measure if:** either zero-action threshold changes.

**History:** The prior feedback run drew log-uniformly over 10–1000 kg,
so it spent little of its budget near the new transition.

## contact_dcm_stability

**Current:** It is a `MetricsTermCfg` with `reduce="mean"`, using exactly
the existing contact-aware DCM callable and parameters. The score is raw,
dimensionless, and independent of reward weight and time-step multiplication.
The safety task replaces its reward with `is_alive` at weight +1.0 while
retaining the previous safety and action penalties.

**Re-measure if:** the contact set or objective callable changes.

**History:** A zero-weight reward would never compute because the reward
manager skips it, so it could not serve as telemetry.

## forward_cart_lag

**Current:** `max(reference_x - cart_x, 0)` in environment-local coordinates,
registered as an episode maximum. The probe traces the signed error too, so
overshoot remains visible. The evaluator records both this maximum and the
failure time, phase, minimum root height, maximum tilt, displacement,
completion and dispatched force before reset. A reactive trigger uses only
the preceding state available before its action: pulse at the first Hold
crossing above 0.05 m for one second, or a 0.05/0.03 m hysteresis gate.
An untriggered action is classified as not measured.
The pulse duration is counted in policy steps, avoiding float32 time
subtraction at the end of the one-second window.

**Re-measure if:** controller dispatch timing or object reference semantics
change.

**History:** The 575 kg trace motivating this probe had lag 0.054 m at
25 s and 0.131 m at 28 s, while a 525 kg trace stayed near 0.022 m.
Those values identify a possible trigger, not an actor observation: cart
pose is available to the probe, not to the current actor.

## measure_responses

**Current:** `scripts/probe_locomanip_reactive.py` ran three repetitions of
all six hand-force channels, both signs and 5/25/50 N, under each schedule
at 525 and 550 kg. Each schedule also ran nine repeated-zero controls.
The frozen profile hash is
`7a48f73834636e851791e26a68d0c5119d77ad29c879c3cdc0f65fe179d954d5`
for the 2026-10-13 run, before inherited path entries were removed;
seed was 42 and encoder bias was off. Raw results and pre-reset traces are
under `logs/locomanip/safety-2026-10-13/authority/`. A physical response
means the last-5-s Hold cart-position RMS difference exceeded the largest
matched repeated-zero difference by 0.01 mm; a target response uses the
analogous controller-q floor plus 1 µrad. These comparisons establish a
measurable effect, not identical simulated trajectories.

| mass | schedule | measured / 108 | untriggered | delivered | cart response | target and cart | observed rescue | failures in requested rows / 108 | zero failures / 9 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 525 kg | constant | 108 | 0 | 108 | 48 | 37 | 0 | 39 | 0 |
| 525 kg | pulse | 0 | 108 | 0 | 0 | 0 | 0 | 0 | 0 |
| 525 kg | reactive | 0 | 108 | 0 | 0 | 0 | 0 | 0 | 0 |
| 550 kg | constant | 108 | 0 | 108 | 41 | 37 | 25 | 83 | 8 |
| 550 kg | pulse | 75 | 33 | 75 | 8 | 7 | 0 | 75 | 7 |
| 550 kg | reactive | 82 | 26 | 82 | 26 | 24 | 1 | 81 | 7 |

All measured inputs matched the native dispatched force columns. The
550 kg pulse first triggered at 29.64–38.34 s and lasted 1.00 s; the
reactive trigger appeared at 28.14–39.70 s and stayed active for
3.50–10.20 s. The first lag crossing often came late in the failing
episode. At 525 kg, neither lag schedule triggered because the zero-action
lag stayed below 0.05 m; those rows provide no authority measurement.
Constant forces could rescue 550 kg but harmed 525 kg in 39/108 cases,
which is why promotion protects every consistently safe mass.

**Re-measure if:** the force routing, Hold timing, controller profile or
failure threshold changes.

## LocomanipSafetyRunner

**Current:** Training requires an `authority_gate.json` from
`scripts/probe_locomanip_reactive.py` via
`MC_MJLAB_SAFETY_AUTHORITY_REPORT`. The gate binds its SHA-256 digest to
the frozen controller profile and requires all six channels in both
directions to reach the controller plus a measurable controller-target and
physical response near the first failing mass. Constant rescue is not
required. The report must state the 525/550 kg transition and explicitly
declare whether encoder bias was used; both settings are accepted. Use
`scripts/train_locomanip_safety.py` to check the report before building
256 controllers. The runner checks it again, copies its exact bytes into
the run directory and embeds its contents and SHA-256 in every checkpoint.
If the force channels do not meet this gate, inspect routing,
timing and coordinated force or walking/cart slowdown before spending the
16.384-million-transition budget.

The training task uses 256 environments, seed 42, 64 rollout steps, two
minibatches, two epochs, 1000 iterations, and saves every 20 iterations.
The registered full task ID is its experiment name. Screen iterations
200, 400, 600, 800 and final with eight pairs per mass at seed 42;
validate the selected checkpoint with 32 pairs per mass at seeds 43 and 44.
`scripts/screen_locomanip_safety.py` discovers the final checkpoint from
the run and checks it before the first screen. It selects only a positive
pooled reduction with complete pairs and no per-mass regression before
starting separate validation.
The fixed grid is 10, 100 and 500–650 kg in 25 kg steps. Its transition
grid for the equal-mass pooled estimate is 500–650 kg. Promotion needs at
least 20% relative combined-failure reduction with a paired 95% interval
excluding zero, plus zero observed policy failures and no new incomplete
episodes at every mass where the baseline safely completes all repetitions.
Any higher failure count at another evaluated mass also blocks promotion.
The evaluator batches up to 192 mixed-mass rows and retries both arms of
an incomplete or infrastructure-affected pair twice with the same input
draws. Persistent missing data remain explicit unresolved conditions and
cannot become successes. Zero
observed failures has finite-sample uncertainty; each mass reports its
Wilson interval. `scripts/evaluate_locomanip_safety.py` writes every
condition and its locked grid before stepping.

**Re-measure if:** the threshold, profile or observation/action interface
changes.

**History:** The prior run improved cart placement while reducing
survival; this gate explicitly prevents a 550 kg improvement from paying
for a new 525 kg failure.

The gated seed-42 run completed 1000 iterations, or 16.384 million
transitions, at
`logs/rsl_rl/Mc-Mjlab-Locomanip-Safety-Locomanipcontroller-Hrp5P-Position-Feedback/2026-10-14_00-14-27`.
The scheduled screens used eight matched pairs per mass, with encoder bias
enabled. The 500 kg baseline had 0/8 combined failures at every screen;
the policy failed there at 3/8, 6/8, 7/8, 8/8 and 8/8 respectively.
At 525 kg the baseline failed 2/8, versus policy failures of 7/8 at
iteration 200 and 8/8 at every later screen. At 550 kg the baseline
failed 4/8, versus 8/8 at every policy screen. The 10 and 100 kg cohorts
remained failure-free; baseline and policy both failed 8/8 at each mass
from 575 to 650 kg. Pooled transition-grid failure was 38/56 for the
baseline. Every checkpoint made it worse:

| iteration | policy failures / 56 | relative reduction | paired 95% interval for absolute reduction |
| ---: | ---: | ---: | ---: |
| 200 | 50 | -31.6% | -30.4 to -12.5 percentage points |
| 400 | 54 | -42.1% | -35.7 to -19.6 percentage points |
| 600 | 55 | -44.7% | -37.5 to -23.2 percentage points |
| 800 | 56 | -47.4% | -39.3 to -25.0 percentage points |
| 999 | 56 | -47.4% | -39.3 to -25.0 percentage points |

All cohorts were complete. No checkpoint passed the safe-mass screen or
the pooled improvement gate, so no off-seed 32-pair validation was run
and no policy was promoted. The reports are under
`logs/locomanip/safety-2026-10-13/screen/`.
