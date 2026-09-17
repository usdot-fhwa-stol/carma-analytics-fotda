# DT-WZ verification analysis

Analyse a pedestrian-detection-to-SDSM verification session: per-run metrics,
an end-to-end latency cascade, and summary tables and plots grouped by
pedestrian dwell condition.

## Running it

```bash
source /home/vmuser/hass_dt_demo/.venv/bin/activate

python src/carma-platform/run_all_dt_wz_analysis.py \
    --data-root /path/to/20260914_verification_test \
    --output-dir out/verification_20260914
```

Roughly 80 s for a 15-run session. Useful flags:

| Flag | Purpose |
| --- | --- |
| `--runs-csv PATH` | Run manifest (default `<data-root>/runs.csv`) |
| `--condition 5sec` | Only one dwell condition |
| `--limit 2` | Only the first N runs, for a quick check |

Requirements: `pandas numpy matplotlib mcap scipy`. **No ROS 2, no `tshark` and
no `pycrate` are needed** — see *Portable backends* below.

```bash
python -m unittest discover -s src/carma-platform/test -v
```

The dataset-dependent tests skip unless the session is on disk; set
`DT_WZ_DATA_ROOT` to point at another one.

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
2026-09-14 session has 20 MCAPs and 19 RSU pcaps for 15 real runs: the 5-second
condition was run twice and only the `_take2` captures and `run7..run11` bags
count, and every bag exists as both a truncated `rosbag2_*` copy (which will not
open) and a readable `recovered_rosbag2_*` one. Nothing in the files themselves
distinguishes the good from the abandoned.

## Outputs

```
<output-dir>/
    analysis_summary.json      pass / fail / N-A / error per metric, plus pooled totals
    summary_by_run.csv         one row per run, every metric as a column
    summary_by_condition.csv   mean and median per dwell condition
    detections_all_runs.csv    every detection, every stage
    qa_report.txt              per-hop coverage, latency, and clock offsets
    stage_summary.csv          the same per-hop table as data
    detections_columns.md      column reference for the detection tables
    plots/
        latency_by_stage_{5sec,10sec,15sec,all}.png
        latency_per_hop_{5sec,10sec,15sec}.png
        latency_cascade_{5sec,10sec,15sec}.png
        latency_by_condition.png
        drop_rates_by_run.png
    <condition>_run<N>/
        stats/*.json  detections.csv  qa_report.txt
```

Re-running over the same inputs reproduces byte-identical CSVs.

## Metrics

| ID | Measures | Passes when |
| --- | --- | --- |
| CP-02 | raw detection → SDSM received at the vehicle | drop rate ≤ 2% |
| CP-03 | RSU broadcast → CARMA Platform receipt | drop rate ≤ 2% |
| CP-04 | detection → Kafka broker accepting it | mean < 0.5 s, late < 2% |
| DT-05 | detection → SDSM receipt at the vehicle | median < 0.3 s |
| PL-01 | per-topic message rates; OBU radio counts | rate within ±20% |

**CP-01** is separate, because it pools across sessions rather than running per
run (the test plan's 30 runs span two recording days):

```bash
python src/carma-platform/run_cp01_analysis.py \
    --data-root .../20260914_verification_test \
    --data-root .../20260915_verification_test \
    --output-dir out/cp01
```

**CS-01** is also separate, and pools every session given into one result:

```bash
python src/carma-platform/run_cs01_analysis.py \
    --data-root .../20260914_verification_test \
    --data-root .../20260915_verification_test \
    --output-dir out/cs01
```

Each session is windowed to its own runs first, then all of them are verified
together, so there is one plot and one set of statistics. Pooling needs the
sessions to share a configured reference, so the references are compared and a
mismatch stops the run rather than averaging two different geometries.

It checks that each SDSM places the spoofed pedestrian at the reference point
FLIRCameraDriver was configured with (mean position error < 0.2 m) and that the
reported heading matches the detection velocity (mean error < 1 deg).

The reference is read from the data, never hard-coded: the detection logs hold
three projection origins about one metre apart, because the driver was
reconfigured on 2026-09-09. The Kafka dumps are also windowed to the session's
runs first, since a dump holds the broker's whole retention and would otherwise
verify several days of testing at once.

It reports "X frames out of 4500 frames dropped", counted at the **camera's
websocket** in the pc2 V2XHub log — the closest measurement point to the camera,
before the plugin parses, queues or forwards anything.

Measuring at the Kafka topic instead would charge the camera for losses that are
not its own: across these 30 runs, 24 frames arrived intact over the websocket
and never reached Kafka, with contiguous `dataNumber` values proving the camera
had sent them. Those are reported separately as `lost_after_camera`.

A stall does not inflate the figure. The count is keyed on the camera's own
capture time, so a frame that arrives late still lands in the dwell window it
belongs to.

Each run's dwell window is found without a recorded entry time, by taking the
detection-burst start whose dwell-length window holds the most frames. Anchoring
on the *first* detection instead would measure a false start (one 2026-09-15 run
opens with an 18-frame burst and a 2.9 s pause before the real dwell), and
anchoring on the *longest burst* would stop at a genuine mid-dwell gap.

Every run is windowed to its **engaged interval**, taken from `/guidance/state`.

A metric reports one of four outcomes, and they are distinct: passed, failed,
**not applicable**, or errored. Not-applicable means the data needed was never
recorded — it is not a failure. MAP and SPAT are in that position for the
2026-09-14 session, whose MCAPs contain neither topic.

## Portable backends

`portable/` reimplements the parts of the stack the metrics actually use, so the
analysis runs on a plain Python venv:

| Module | Replaces | Note |
| --- | --- | --- |
| `mcap_backend` | `rosbag2_py` / `rclpy` | decodes CDR from the ros2msg schema text stored inside the MCAP, so no generated message packages are needed |
| `pcap_backend` | `tshark` + `pycrate` | timestamps come from the pcap record header; correlation is on payload bytes, so no ASN.1 decode is required |
| `kafka_log` | — | tolerates both tab- and pipe-delimited console dumps |
| `tcpdump_text` | — | reads OBU captures saved as tcpdump console text |
| `obu_capture` | — | dispatches on the OBU file's actual format, binary or text |

`parse_ros2_bags` falls back to `mcap_backend` automatically when `rosbag2_py`
is absent, so the other analyses in this directory gain the same portability.

## Reading the results carefully

Three properties of this data change what the numbers mean.

**`t_rsu_broadcast` is the RSU's transmit instant.** This session captured the
broadcasting RSU; earlier sessions captured the OBU, where the equivalent stage
(`t_ota_capture`) sits one propagation hop later. End-to-end totals are
therefore slightly smaller than the 2026-09-11 figures for a reason that has
nothing to do with the system getting faster.

**`t_obu_radio_rx` is only as strong as the OBU capture format.** Both forms are
in use and both are named `.pcap`, so the reader decides from content:

* **binary pcap** (2026-09-15 on) carries payload bytes, so the stage is joined
  on exact identity like every other stage from `t_streets_encode` onward, and
  a true over-the-air reception rate is reported (`ota_reception_rate_pct`).
* **tcpdump text** (2026-09-14) carries no payload, so the stage falls back to
  nearest-time matching within 60 ms. Sound in aggregate, not per row.

`summary_by_run.csv` records which was used in `obu_payload_matched`. Check it
before comparing this stage across sessions.

**Cross-host hops can come out negative, and that is a clock offset.** A message
cannot arrive before it was sent, so a persistently negative hop measures the
offset between two hosts' clocks. `qa_report.txt` estimates and reports these;
none are silently corrected, because clamping one to zero would hide a real
finding and shift the neighbouring hop by the same amount.
