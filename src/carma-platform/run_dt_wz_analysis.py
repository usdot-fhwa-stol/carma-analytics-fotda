#!/usr/bin/env python3
"""Run one DT-WZ analysis. See ``dt_wz_analysis_util.run_dt_wz_analysis``.

    python src/carma-platform/run_dt_wz_analysis.py --list
    python src/carma-platform/run_dt_wz_analysis.py cp02 \\
        --data-root .../20260914_verification_test \\
        --output-dir out/cp02

A launcher. It puts this directory on ``sys.path`` so the package imports
whether or not the working directory happens to be this one, then hands over.
Everything the command does is in the package module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dt_wz_analysis_util.run_dt_wz_analysis import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
