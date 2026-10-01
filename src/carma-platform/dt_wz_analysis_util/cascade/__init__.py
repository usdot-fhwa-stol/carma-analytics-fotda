"""End-to-end SDSM latency cascade: camera detection to fused output at the vehicle.

Produces one row per FLIR camera detection carrying an absolute timestamp for
every stage it reached, so a per-hop latency breakdown and an end-to-end total
fall out as column differences. See ``cascade_config`` for the stage list and for
the two stages whose meaning changed in this session.
"""

from . import build, cascade_config, parse_sdss, parse_v2xhub

__all__ = ["build", "cascade_config", "parse_sdss", "parse_v2xhub"]
