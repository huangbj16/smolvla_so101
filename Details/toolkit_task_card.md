# Task card: toolkit assembly, cylinder into fixture (real track)

Written 2026-09-17. Linked from [08](../08_data_quality_research.md) §A.4. Used for Phase 0.5 and the
real-track dials. Items marked *default* are my suggestions; edit them before recording, then keep them
fixed.

## Task

Pick up the **white cylinder** from the table and place it in the **top-left hole** of the **black fixture**.

Task string (use verbatim in every recording):
`Pick up the white cylinder and place it in the top-left hole of the black fixture`

## Success

| | Definition |
|---|---|
| **Binary** | Within **30 s** of episode start, the cylinder is **fully seated** in the top-left hole and the gripper has released it |
| **Fully seated** | *Default:* cylinder upright, its top at or below the hole's rim, and it stays put after release. Must be checkable from video by a blind rater |
| **Top-left** | *Default:* as seen in the **top camera image**. Mark that hole on the fixture with tape |
| **Stages** | reached (0.2) · grasped + lifted (0.4) · over the hole (0.6) · inserted (0.8) · fully seated + released (1.0) |

## What varies and what's fixed

| Factor | Decision |
|---|---|
| Cylinder **position** | **VARIES** over **10 positions (P1–P10)** spread evenly over the reachable table area, each marked with tape. *Default:* 2 rows × 5, keeping clear of the fixture so the cylinder never blocks the target |
| Cylinder orientation | Fixed: always standing upright |
| Fixture position | Fixed: outline taped or screwed down; the other holes stay empty |
| Cylinder instance | Fixed: one cylinder (write its diameter/height and the hole clearance here: ___) |
| Scene | Fixed: table surface, lighting on, camera mounts, Cameractrls presets 1 (C920) / 2 (C922) |

## Episode protocol

- **Start:** arm at the home pose, gripper open, cylinder standing on the scheduled position.
- **Clean demo (D0):** one grasp style (*default:* straight down from above), one approach path, one-shot
  grasp, no pauses. *Default:* small wiggles during insertion are allowed; a regrasp is not (that's D3).
- **End:** after release, return the arm to home, then press **→** to end the episode. The recording
  stops at 30 s regardless.
- **Failed demo:** press **←** to re-record it right away. For dial data later, keep failures and log
  them instead.
- **Reset:** place the cylinder on the next scheduled position during the reset window (15 s).

## Phase 0.5 schedule: 50 episodes

- **5 rounds × 10 positions.** Each round visits P1–P10 once, in a shuffled order written down before
  recording. This spreads any drift over the day (lighting, fatigue) evenly across positions.
- One `lerobot-record` run per round (`num_episodes=10`), so episode indices line up with the list.
- Check with the first round that the cylinder stays visible in the wrist camera during the approach.

## Per-episode log

Keep `phase05_log.csv` next to the dataset, one row per episode:

`episode_index, round, position (P1–P10), success (0/1), stage, dial (D0), notes`

The later analyses need this, and it can't be reconstructed afterwards.
