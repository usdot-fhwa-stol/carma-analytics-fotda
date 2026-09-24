"""Support library for the DT-WZ verification analyses.

Everything the analyses need lives here. There is one entry point, and the
tests differ from each other in data rather than in code.

```
dt_wz_analysis_util/
    config.py     the test catalogue and every setting specific to it --
                  limits, tolerances, site geometry, topic names, plot style
    run_dt_wz_analysis.py
                  the single entry point: one handler per measurement, and a
                  command line generated from each test's settings
    dataset.py    runs.csv and session layout -- what constitutes a run
    metrics.py    the per-run measurements shared by several tests
    report.py     windowing, pooling, output files and plots shared by the tests
    camera_detections.py   camera frame drops, measured at the websocket
    message_rates.py       per-topic message communication rates
    vehicle_yield.py       whether the vehicle stopped short of the pedestrian
    location_spoofing.py   that SDSMs place the object at the configured reference
    readers/      format readers: MCAP, pcap, Kafka dumps, tcpdump text
    timeutil.py   timestamp normalisation, shared by the readers and parsers
    pairing.py    greedy forward pairing for log lines that carry no key
    cascade/      the end-to-end per-detection latency table and its plots
```

The division of labour is the point. Each measurement module is named for what
it measures and takes every limit, tolerance, coordinate and output name as an
argument, so it holds no test-specific values and does not know which test uses
it. ``config.py`` holds the test catalogue and the values;
``run_dt_wz_analysis.py`` reads the catalogue and passes them in. A value can
therefore be changed in one place, and the value a report states is by
construction the value the measurement applied.

The package sits beside the repo's own ``carma-platform`` modules (``utils``,
``guidance_scripts``, ``parse_ros2_bags``) and imports them by plain name, so
the parent directory is put on ``sys.path`` here. That keeps the analyses
runnable from any working directory instead of only from their own.
"""

import sys
from pathlib import Path

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

__all__ = [
    "camera_detections", "cascade", "config", "dataset", "location_spoofing",
    "message_rates", "metrics", "pairing", "readers", "report",
    "run_dt_wz_analysis", "timeutil", "vehicle_yield",
]
