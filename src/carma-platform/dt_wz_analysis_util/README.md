# DT-WZ verification analysis

Analyses a pedestrian-detection-to-SDSM verification session.

Requires `pandas numpy matplotlib mcap scipy`. No ROS 2, no `tshark`, no
`pycrate`.

## Running it

List the tests:

```bash
python src/carma-platform/run_dt_wz_analysis.py --list
```

Run one. Repeat `--data-root` to pool several sessions into one result:

```bash
python src/carma-platform/run_dt_wz_analysis.py cp02 \
    --data-root <session-dir> \
    --data-root <another-session-dir> \
    --output-dir out/cp02
```

Run them all:

```bash
python src/carma-platform/run_all_dt_wz_analysis.py \
    --data-root <session-dir> \
    --data-root <another-session-dir> \
    --pl01-data-root <message-rate-session-dir> \
    --output-dir out
```

Add `--only cp02 cp03` to run a subset.

PL-01 needs two groups of sessions, because no single session carries every
message type. `--pl01-data-root` supplies the second group to the wrapper;
`--secondary-data-root` does the same for the single-test entry point.

Every limit and tolerance is a field in `config.py` and has a generated flag:

```bash
python src/carma-platform/run_dt_wz_analysis.py cp02 --help
```

Tests:

```bash
python -m pytest src/carma-platform/test -q
```

## Expected input

One directory per session:

```
<data-root>/
    runs.csv                                   the manifest
    rsu_pcap/       capture_*.pcap             SDSM broadcasts (binary pcap)
    obu/            <cond>-run<N>.pcap         OBU radio (binary pcap OR tcpdump text)
    rosbags/        recovered_rosbag2_*.mcap   CARMA Platform recordings
    pc1/            v2xhub_pc1_*.log
    pc2/            v2xhub_pc2_*.log, sdss_*.log, kafka_topics_*/*.log
```

These directory names and glob patterns are `config.SessionLayout`, so a
differently arranged session needs a configuration change, not a code change.

`runs.csv` lists one row per run and is the authority on which files belong to
it — the tool never globs for recordings. Columns:

```
run_condition, run_id, start_time_etc, rsu_pcap_fn, obu_pcap_fn, rosbag_fn
```

`start_time_etc` is America/New_York wall clock. A session that records no RSU
capture and needs no start time may omit those two columns.
