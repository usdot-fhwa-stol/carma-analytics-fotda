# DT-WZ verification analysis

Analyses a pedestrian-detection-to-SDSM verification session: per-metric results,
an end-to-end latency cascade, and tables and plots grouped by dwell condition.

## Running it

Each metric has its own script, and they are the real entry points:

```bash
source /home/vmuser/hass_dt_demo/.venv/bin/activate

python src/carma-platform/run_cp02_analysis.py \
    --data-root /path/to/20260914_verification_test \
    --data-root /path/to/20260915_verification_test \
    --output-dir out/cp02
```

Every script takes the same two arguments. Repeat `--data-root` to pool sessions
into one result; each session is windowed to its own runs first.

**PL-01 takes two groups of sessions**, because no single session carries every
message: MAP and SPAT were not broadcast during the verification runs, and the
session recorded to exercise them carries no SDSM.

```bash
python src/carma-platform/run_pl01_analysis.py \
    --data-root .../20260917 \
    --sdsm-data-root .../20260914_verification_test \
    --sdsm-data-root .../20260915_verification_test \
    --output-dir out/pl01
```

Two topics are averaged over the periods when they were present rather than over
the whole engaged window, because neither runs continuously: **SDSM** flows only
while an object is detected, and **MOM** stops when the vehicle drives out of
range of the source. Averaging either over the whole window measures how long it
was absent rather than how fast it ran.

`run_all_dt_wz_analysis.py` is a wrapper that runs them all in turn:

```bash
python src/carma-platform/run_all_dt_wz_analysis.py \
    --data-root .../20260914_verification_test \
    --data-root .../20260915_verification_test \
    --pl01-data-root .../20260917 \
    --output-dir out
```

`--pl01-data-root` routes PL-01's two groups; without it every metric uses
`--data-root`. About 5.5 minutes for 30 runs across two sessions. Use `--only cp02 cp03` to run
a subset. Prefer the individual scripts when re-running one metric: CP-02 and the
cascade each parse logs of over a million lines.

Requirements: `pandas numpy matplotlib mcap scipy`. **No ROS 2, no `tshark` and
no `pycrate`** — see *Portable backends*.

```bash
python -m pytest src/carma-platform/test -q
```

## Layout

```
src/carma-platform/
    run_cp01_analysis.py ... run_pl01_analysis.py   entry points, one per metric
    run_cascade_analysis.py                         latency breakdown
    run_all_dt_wz_analysis.py                       wrapper over the above
    dt_wz_analysis_util/                            everything they depend on
        dataset.py    runs.csv and session layout
        metrics.py    the per-run measurements
        report.py     windowing, pooling, output files, plots
        cp01.py       camera detection drops
        cs01.py       location spoofing verification
        readers/      MCAP, pcap, Kafka dump and tcpdump-text readers
        timeutil.py   timestamp normalisation
        pairing.py    pairing for log lines with no correlatable key
        cascade/      per-detection stage table and its plots
```

## Metrics

| Script | Measures | Passes when |
| --- | --- | --- |
| `run_cp01_analysis.py` | camera frames missing from a detection burst | reported, no limit |
| `run_cp02_analysis.py` | detections that never reached the vehicle | drop rate <= 2% |
| `run_cp03_analysis.py` | RSU broadcasts the vehicle never received | drop rate <= 2% |
| `run_dt05_analysis.py` | camera detection to SDSM at the vehicle | median < 0.3 s |
| `run_pl01_analysis.py` | per-topic message rates, OBU radio activity | rate within +/-20% |

| `run_cs01_analysis.py` | SDSM places the pedestrian at the reference | < 0.2 m, < 1 deg |
| `run_pl03_analysis.py` | vehicle yields to the pedestrian it was warned about | >= 90% of valid runs |
| `run_cascade_analysis.py` | per-hop latency, camera to fused output | reported, no limit |

Every run is windowed to its **engaged interval**, taken from `/guidance/state`.

Each drop-rate metric reports two figures, because they answer different
questions and can disagree: the **pooled rate** (how much was lost overall) and
the **per-run pass rate** (how often the limit was met). One bad run can fail on
its own while barely moving the pooled rate.

A metric reports one of four outcomes, and they are distinct: passed, failed,
**not applicable**, or errored. Not-applicable means the data needed was never
recorded. MAP and SPAT are in that position for these sessions.

## Expected session layout

```
<data-root>/
    runs.csv                                   the manifest; the authority
    rsu_pcap/       capture_*.pcap             SDSM broadcasts (binary pcap)
    obu/            <cond>-run<N>.pcap         OBU radio (binary pcap OR tcpdump text)
    rosbags/        recovered_rosbag2_*.mcap   CARMA Platform recordings
    pc1/            v2xhub_pc1_*.log
    pc2/            v2xhub_pc2_*.log, sdss_*.log, kafka_topics_*/*.log
```

`runs.csv` columns: `run_condition, run_id, start_time_etc, rsu_pcap_fn,
obu_pcap_fn, rosbag_fn`. Times are America/New_York wall clock.

**`runs.csv` decides what is a run — the tool never globs for recordings.** The
2026-09-14 session holds 20 MCAPs and 19 RSU pcaps for 15 real runs: the 5-second
condition was run twice, and every bag exists as both a truncated `rosbag2_*`
copy (which will not open) and a readable `recovered_rosbag2_*` one. Nothing in
the files themselves separates the good from the abandoned.

## Format readers

`dt_wz_analysis_util/readers/` reimplements the parts of the stack the metrics
actually use, so the analysis runs on a plain Python venv:

| Module | Replaces | Note |
| --- | --- | --- |
| `mcap_reader` | `rosbag2_py` / `rclpy` | decodes CDR from the ros2msg schema text stored inside the MCAP, so no generated message packages are needed |
| `pcap_reader` | `tshark` + `pycrate` | timestamps come from the pcap record header; correlation is on payload bytes, so no ASN.1 decode is required |
| `kafka_log` | — | tolerates both tab- and pipe-delimited console dumps |
| `tcpdump_text` | — | reads OBU captures saved as tcpdump console text |
| `obu_capture` | — | dispatches on the OBU file's actual format, binary or text |
| `flir_websocket` | — | the camera's raw websocket stream from the pc2 V2XHub log |

`parse_ros2_bags` falls back to `mcap_reader` automatically when `rosbag2_py` is
absent, so the other analyses in this directory gain the same portability.

**PL-03 caches its geometry.** Each run's trajectory and pedestrian points are
written to `pl03_tracks.npz`, so `--plot-only` redraws both figures in about a
second instead of re-reading 30 recordings. If the cache is absent it runs the
full analysis once and builds it, so the flag is always safe to pass.

**PL-03 scores only the valid runs.** A run counts toward the yield rate only if
its camera detections were consistent and its SDSMs were consistently received,
so the metric measures vehicle behaviour rather than sensing faults. Both checks
are applied and every excluded run records which one excluded it. The trial is
bounded by the engaged window; the start and end points confirm the vehicle drove
the intended route, with the end checked over the whole recording since CARMA
normally disengages once past the pedestrian.

## Reading the results carefully

**CP-01 is measured at the camera's websocket, against each run's own burst.**
The websocket is the closest point to the camera, before the plugin parses or
queues anything; counting at Kafka instead charges the camera for 24 frames the
plugin lost. And the pedestrian never stood in the zone for exactly the labelled
time — one "15 s" run lasts 12.71 s — so each run is measured against the frames
a continuous stream would hold over its own burst span. Counting against the
nominal dwell measures the pedestrian's timing, not the camera.

**`t_rsu_broadcast` is the RSU's transmit instant.** These sessions captured the
broadcasting RSU; earlier ones captured the OBU, where the equivalent stage sat
one propagation hop later. End-to-end totals are not directly comparable.

**`t_obu_radio_rx` is only as strong as the OBU capture format.** A binary pcap
carries payloads, so the stage is joined on exact identity and a true
over-the-air reception rate is reported. A tcpdump text capture carries none, so
the stage falls back to nearest-time matching. `obu_payload_matched` records
which was used.

**Cross-host hops can come out negative, and that is a clock offset.** A message
cannot arrive before it was sent, so a persistently negative hop measures the
offset between two hosts' clocks. `cascade/qa_report.txt` estimates and reports
these; none are silently corrected, because clamping one to zero would hide a
real finding and shift the neighbouring hop by the same amount.
