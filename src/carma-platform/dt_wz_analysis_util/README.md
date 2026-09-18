# DT-WZ verification analysis

Analyses a pedestrian-detection-to-SDSM verification session: per-test results,
an end-to-end latency cascade, and tables and plots grouped by dwell condition.

## Running it

There is one entry point. It sits in the directory above this package, beside
`run_all_dt_wz_analysis.py`. Run it from the repository root:

```bash
python src/carma-platform/run_dt_wz_analysis.py --list
```

That prints every test, what it measures and the rule it passes by. To run one:

```bash
python src/carma-platform/run_dt_wz_analysis.py cp02 \
    --data-root <session-dir> \
    --data-root <another-session-dir> \
    --output-dir out/cp02
```

Every test takes the same two arguments. Repeat `--data-root` to pool sessions
into one result; each session is windowed to its own runs first.

`run_all_dt_wz_analysis.py` runs them all in turn:

```bash
python src/carma-platform/run_all_dt_wz_analysis.py \
    --data-root <session-dir> \
    --data-root <another-session-dir> \
    --pl01-data-root <message-rate-session-dir> \
    --output-dir out
```

About 5.5 minutes for 30 runs across two sessions. Use `--only cp02 cp03` to run
a subset. Prefer the single-test entry point when re-running one test: CP-02 and
the cascade each parse logs of over a million lines.

Requirements: `pandas numpy matplotlib mcap scipy`. **No ROS 2, no `tshark` and
no `pycrate`** — see *Format readers*.

```bash
python -m pytest src/carma-platform/test -q
```

### Two groups of sessions for PL-01

No single session carries every message type: MAP and SPAT were not broadcast
during the verification runs, and the session recorded to exercise them carries
no SDSM. So PL-01 takes a second group of sessions:

```bash
python src/carma-platform/run_dt_wz_analysis.py pl01 \
    --data-root <message-rate-session-dir> \
    --secondary-data-root <verification-session-dir> \
    --output-dir out/pl01
```

`--pl01-data-root` on the wrapper routes the same two groups. Without either,
every topic is measured on `--data-root`.

Two topics are averaged over the periods when they were present rather than over
the whole engaged window, because neither runs continuously: **SDSM** flows only
while an object is detected, and **MOM** stops when the vehicle drives out of
range of the source. Averaging either over the whole window measures how long it
was absent rather than how fast it ran.

### Changing a limit or a tolerance

Every acceptance limit and tunable is a field of a settings dataclass in
[`config.py`](config.py), and **the command line is generated from those
fields**. So each one has a flag, and `--help` on a test lists exactly what that
test can be re-run with:

```bash
python src/carma-platform/run_dt_wz_analysis.py cp02 --help
python src/carma-platform/run_dt_wz_analysis.py cp02 ... --max-drop-rate-pct 5
```

A value given this way reaches the measurement, not only the report. Adding a
field to a settings class adds its flag; no parser is edited.

## Layout

```
src/carma-platform/
    run_dt_wz_analysis.py       launcher for one test
    run_all_dt_wz_analysis.py   launcher for all of them
    dt_wz_analysis_util/        everything they depend on
        config.py               the test catalogue and every setting in it
        run_dt_wz_analysis.py   one handler per measurement, generated CLI
        dataset.py              runs.csv and session layout
        metrics.py              per-run measurements shared by several tests
        report.py               windowing, pooling, output files, plots
        camera_detections.py    camera frame drops at the websocket
        message_rates.py        per-topic message communication rates
        vehicle_yield.py        did the vehicle stop short of the pedestrian
        location_spoofing.py    SDSM places the object at the reference
        readers/                MCAP, pcap, Kafka dump, tcpdump-text readers
        timeutil.py             timestamp normalisation
        pairing.py              pairing for log lines with no correlatable key
        cascade/                per-detection stage table and its plots
```

**The measurement modules are named for what they measure, not for the test that
asks for it.** None of them contains a test code, a threshold or an output file
name: each takes those as arguments. `config.py` holds the catalogue of tests
and all their values, and the entry point passes them in. A limit can therefore
be changed in one place, and the limit a report states is by construction the
limit the measurement applied.

## Tests

| Test | Measures | Passes when |
| --- | --- | --- |
| `cp01` | camera frames missing from a detection burst | reported, no limit |
| `cp02` | detections that never reached the vehicle | drop rate <= 2% |
| `cp03` | RSU broadcasts the vehicle never received | drop rate <= 2% |
| `dt05` | camera detection to SDSM at the vehicle | median < 0.3 s |
| `pl01` | per-topic message rates, OBU radio activity | rate within +/-20% |
| `pl03` | vehicle yields to the pedestrian it was warned about | >= 90% of valid runs |
| `cs01` | SDSM places the pedestrian at the reference | < 0.2 m, < 1 deg |
| `cascade` | per-hop latency, camera to fused output | reported, no limit |

Every run is windowed to its **engaged interval**, taken from `/guidance/state`.

Each drop-rate test reports two figures, because they answer different questions
and can disagree: the **pooled rate** (how much was lost overall) and the
**per-run pass rate** (how often the limit was met). One bad run can fail on its
own while barely moving the pooled rate.

A test reports one of four outcomes, and they are distinct: passed, failed,
**not applicable**, or errored. Not-applicable means the data needed was never
recorded. MAP and SPAT are in that position for the verification sessions.

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

These directory names and glob patterns are `config.SessionLayout`, so a
differently arranged session is a configuration change rather than a code one.

`runs.csv` columns: `run_condition, run_id, start_time_etc, rsu_pcap_fn,
obu_pcap_fn, rosbag_fn`. Times are America/New_York wall clock.

**`runs.csv` decides what is a run — the tool never globs for recordings.** The
2026-09-14 session holds 20 MCAPs and 19 RSU pcaps for 15 real runs: the 5-second
condition was run twice, and every bag exists as both a truncated `rosbag2_*`
copy (which will not open) and a readable `recovered_rosbag2_*` one. Nothing in
the files themselves separates the good from the abandoned.

## Format readers

`readers/` reimplements the parts of the stack the measurements actually use, so
the analysis runs on a plain Python venv:

| Module | Replaces | Note |
| --- | --- | --- |
| `mcap_reader` | `rosbag2_py` / `rclpy` | decodes CDR from the ros2msg schema text stored inside the MCAP, so no generated message packages are needed |
| `pcap_reader` | `tshark` + `pycrate` | timestamps come from the pcap record header; correlation is on payload bytes, so no ASN.1 decode is required |
| `kafka_log` | — | tolerates both tab- and pipe-delimited console dumps |
| `tcpdump_text` | — | reads OBU captures saved as tcpdump console text |
| `obu_capture` | — | dispatches on the OBU file's actual format, binary or text |
| `flir_websocket` | — | the camera's raw websocket stream from the pc2 V2XHub log |

`parse_ros2_bags` falls back to `mcap_reader` automatically when `rosbag2_py` is
absent, so the other analyses in the parent directory gain the same portability.

## Reading the results carefully

**CP-01 is measured at the camera's websocket, against each run's own burst.**
The websocket is the closest point to the camera, before the plugin parses or
queues anything; counting at Kafka instead charges the camera for 24 frames the
plugin lost. And the pedestrian never stood in the zone for exactly the labelled
time — one "15 s" run lasts 12.71 s — so each run is measured against the frames
a continuous stream would hold over its own burst span. Counting against the
nominal dwell measures the pedestrian's timing, not the camera.

**PL-03 scores only the valid runs.** A run counts toward the yield rate only if
its camera detections were consistent and its SDSMs were consistently received,
so the test measures vehicle behaviour rather than sensing faults. Both checks
are applied and every excluded run records which one excluded it. The trial is
bounded by the engaged window; the start and end points confirm the vehicle drove
the intended route, with the end checked over the whole recording since CARMA
normally disengages once past the pedestrian.

**PL-03 caches its geometry.** Each run's trajectory and pedestrian points are
written to the cache named in `config.VehicleYieldConfig`, so `--plot-only`
redraws the figures in about a second instead of re-reading 30 recordings. If the
cache is absent it runs the full analysis once and builds it, so the flag is
always safe to pass.

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
