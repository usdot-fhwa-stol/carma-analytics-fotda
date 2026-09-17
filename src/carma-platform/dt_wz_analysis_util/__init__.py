"""Support library for the DT-WZ verification analyses.

Everything the ``run_*_analysis.py`` scripts need lives here, so that directory
holds only entry points. The scripts stay thin: each one parses its arguments,
names a measurement function, and hands the rest to this package.

```
dt_wz_analysis_util/
    dataset.py    runs.csv and session layout -- what constitutes a run
    metrics.py    the per-run measurements (CP-02, CP-03, CP-04, DT-05, PL-01)
    report.py     windowing, pooling, output files and plots shared by the scripts
    cp01.py       camera detection drops, measured at the websocket
    cs01.py       SDSM location spoofing verification
    portable/     ROS-free readers for MCAP, pcap, Kafka and tcpdump captures
    cascade/      the end-to-end per-detection latency table and its plots
```

The package sits beside the repo's own ``carma-platform`` modules (``utils``,
``guidance_scripts``, ``parse_ros2_bags``) and imports them by plain name, so the
parent directory is put on ``sys.path`` here. That keeps the scripts runnable
from any working directory instead of only from their own.
"""

import sys
from pathlib import Path

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

__all__ = ["cascade", "cp01", "cs01", "dataset", "metrics", "portable", "report"]
