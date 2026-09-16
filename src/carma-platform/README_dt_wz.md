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
    obu/hass-wz-logs/  <cond>-run<N>.pcap      OBU radio (tcpdump TEXT)
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

`parse_ros2_bags` falls back to `mcap_backend` automatically when `rosbag2_py`
is absent, so the other analyses in this directory gain the same portability.

## Reading the results carefully

Three properties of this data change what the numbers mean.

**`t_rsu_broadcast` is the RSU's transmit instant.** This session captured the
broadcasting RSU; earlier sessions captured the OBU, where the equivalent stage
(`t_ota_capture`) sits one propagation hop later. End-to-end totals are
therefore slightly smaller than the 2026-09-11 figures for a reason that has
nothing to do with the system getting faster.

**`t_obu_radio_rx` is matched by time, not by payload.** The OBU capture is
tcpdump text and carries no payload bytes, so this stage is matched to the
nearest broadcast within 60 ms. It is sound in aggregate and should not be
trusted for an individual row. Every other stage from `t_streets_encode` onward
is joined on exact payload bytes.

**Cross-host hops can come out negative, and that is a clock offset.** A message
cannot arrive before it was sent, so a persistently negative hop measures the
offset between two hosts' clocks. `qa_report.txt` estimates and reports these;
none are silently corrected, because clamping one to zero would hide a real
finding and shift the neighbouring hop by the same amount.
