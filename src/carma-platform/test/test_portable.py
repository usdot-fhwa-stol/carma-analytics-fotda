"""Tests for the portable backends and the DT-WZ dataset spec.

The parser tests are self-contained and always run. The dataset tests need the
2026-09-14 verification session on disk and skip cleanly without it; point
``DT_WZ_DATA_ROOT`` at another session to run them elsewhere.

The golden-run figures in ``TestGoldenRun`` were verified by hand against four
independent sources before being written down, and they are what catches a
regression in windowing or payload matching -- both of which fail by producing
plausible-looking numbers rather than by raising.
"""

import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dt_wz_analysis_util import cp01  # noqa: E402
from dt_wz_analysis_util import cs01  # noqa: E402
from dt_wz_analysis_util import pl03  # noqa: E402
from dt_wz_analysis_util.portable import (  # noqa: E402
    kafka_log, obu_capture, pcap_backend, tcpdump_text,
)

DATA_ROOT = Path(
    os.environ.get(
        "DT_WZ_DATA_ROOT",
        "/home/vmuser/hass_dt_demo/HASS_DT_Workzone_Data_Logs/20260914_verification_test",
    )
)
HAS_DATA = (DATA_ROOT / "runs.csv").is_file()


class TestKafkaLogParsing(unittest.TestCase):
    """The delimiter bug this fixes corrupted timestamps silently, not loudly."""

    TAB_LINE = 'CreateTime:1757520202123\t{"objectId":1,"timestamp":1757520202100}\n'
    PIPE_LINE = (
        'CreateTime:1788530425666|Partition:0|Offset:46063|null|'
        '{"objectId":1177,"timestamp":1788530425620}\n'
    )

    def _parse(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
            handle.write(text)
            path = handle.name
        try:
            return kafka_log.parse_kafka_log_records(path)
        finally:
            os.unlink(path)

    def test_tab_delimited(self):
        records = self._parse(self.TAB_LINE)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["create_time_ms"], 1757520202123)
        self.assertEqual(records[0]["objectId"], 1)

    def test_pipe_delimited(self):
        records = self._parse(self.PIPE_LINE)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["create_time_ms"], 1788530425666)
        self.assertEqual(records[0]["objectId"], 1177)

    def test_timestamp_is_13_digits_not_concatenated(self):
        """The original bug produced a 36-digit value from the pipe form."""
        for line in (self.TAB_LINE, self.PIPE_LINE):
            record = self._parse(line)[0]
            self.assertEqual(
                len(str(record["create_time_ms"])), 13,
                "create_time_ms must be epoch milliseconds, not concatenated digits",
            )

    def test_mixed_delimiters_in_one_file(self):
        records = self._parse(self.TAB_LINE + self.PIPE_LINE)
        self.assertEqual([r["create_time_ms"] for r in records],
                         [1757520202123, 1788530425666])

    def test_multiline_body_is_reassembled(self):
        text = 'CreateTime:1788530425666|Partition:0|null|{"objectId":5,\n"timestamp":1788530425620}\n'
        records = self._parse(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["objectId"], 5)

    def test_undecodable_body_is_skipped_not_raised(self):
        records = self._parse('CreateTime:1788530425666|null|{"broken":\n')
        self.assertEqual(records, [])

    def test_window_records_filters_by_timestamp(self):
        records = [
            {"timestamp": 1000.0, "create_time_ms": 1000},
            {"timestamp": 2000.0, "create_time_ms": 2000},
            {"timestamp": 3000.0, "create_time_ms": 3000},
        ]
        self.assertEqual(len(kafka_log.window_records(records, 1500, 2500)), 1)


class TestTcpdumpText(unittest.TestCase):
    BSM = "18:51:52.671839 IP6 80f8:f80::2a8.2600 > ff02::1.2600: UDP, length 48\n"
    SDSM = "18:51:52.973719 IP6 80f8:f80::2a8.9000 > 80f8:f80::2a8.9000: UDP, length 62\n"

    def _parse(self, text, day=date(2026, 9, 14)):
        with tempfile.NamedTemporaryFile("w", suffix=".pcap", delete=False) as handle:
            handle.write(text)
            path = handle.name
        try:
            return tcpdump_text.parse_tcpdump_text(path, day)
        finally:
            os.unlink(path)

    def test_classifies_bsm_and_sdsm(self):
        packets = self._parse(self.BSM + self.SDSM)
        self.assertEqual([p["msg_type"] for p in packets], ["BSM", "SDSM"])
        self.assertEqual([p["direction"] for p in packets], ["outgoing", "incoming"])

    def test_timestamp_is_utc_on_the_given_date(self):
        packet = self._parse(self.BSM)[0]
        expected = datetime(2026, 9, 14, 18, 51, 52, 671839, tzinfo=timezone.utc)
        self.assertAlmostEqual(packet["timestamp"], expected.timestamp(), places=4)

    def test_midnight_rollover_advances_the_date(self):
        text = (
            "23:59:59.900000 IP6 a::1.2600 > ff02::1.2600: UDP, length 48\n"
            "00:00:00.100000 IP6 a::1.2600 > ff02::1.2600: UDP, length 48\n"
        )
        first, second = self._parse(text)
        gap = second["timestamp"] - first["timestamp"]
        self.assertAlmostEqual(gap, 0.2, places=4,
                               msg="a capture crossing midnight must not jump back a day")

    def test_counts_and_timestamps_filter_by_window(self):
        packets = self._parse(self.BSM + self.SDSM)
        self.assertEqual(tcpdump_text.count_by_type(packets), {"BSM": 1, "SDSM": 1})
        window = (packets[1]["timestamp"] - 0.001, packets[1]["timestamp"] + 0.001)
        self.assertEqual(
            tcpdump_text.count_by_type(packets, *window), {"SDSM": 1}
        )

    def test_non_udp_lines_are_ignored(self):
        self.assertEqual(self._parse("garbage\n\n18:00:00 something else\n"), [])


class TestPcapDefiniteLength(unittest.TestCase):
    """Long-form ASN.1 lengths; the SDSM-only predecessor handled short form only."""

    def test_short_form(self):
        self.assertEqual(pcap_backend._definite_length(b"\x3c", 0), (0x3C, 1))

    def test_long_form_one_byte(self):
        self.assertEqual(pcap_backend._definite_length(b"\x81\xc8", 0), (200, 2))

    def test_long_form_two_bytes(self):
        self.assertEqual(pcap_backend._definite_length(b"\x82\x01\x00", 0), (256, 3))

    def test_rejects_out_of_range(self):
        self.assertEqual(pcap_backend._definite_length(b"", 0), (None, 0))
        self.assertEqual(pcap_backend._definite_length(b"\x84\x01", 0), (None, 0))

    def test_find_message_requires_end_alignment(self):
        ids = {b"\x00\x29": "SDSM"}
        packet = b"\xff\xff" + b"\x00\x29\x03" + b"abc"
        self.assertEqual(pcap_backend._find_message(packet, ids)[0], "SDSM")
        # Same id, but the declared length does not reach the end of the packet.
        self.assertIsNone(pcap_backend._find_message(packet + b"tail", ids)[0])


class TestObuCaptureDispatch(unittest.TestCase):
    """The OBU capture is binary pcap in some sessions and tcpdump text in others.

    Both are named ``.pcap``, so the reader must decide from content. Getting this
    wrong is silent: the text parser finds no matching lines in a binary file and
    returns an empty list, which looks like a run where the radio heard nothing.
    """

    TEXT = "18:51:52.671839 IP6 a::1.2600 > ff02::1.2600: UDP, length 48\n"

    def test_text_capture_reports_no_payloads(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pcap", delete=False) as handle:
            handle.write(self.TEXT)
            path = handle.name
        try:
            capture = obu_capture.read_obu_capture(path, date(2026, 9, 14))
            self.assertFalse(capture["payloads_available"])
            self.assertEqual(len(capture["messages"]), 1)
            self.assertIsNone(capture["messages"][0]["payload_hex"])
        finally:
            os.unlink(path)

    def test_text_capture_without_a_date_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pcap", delete=False) as handle:
            handle.write(self.TEXT)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                obu_capture.read_obu_capture(path, None)
        finally:
            os.unlink(path)


class TestCp01Anchoring(unittest.TestCase):
    """CP-01 has to find the dwell window without a recorded entry time.

    Two failure modes, both of which produced badly wrong numbers in development
    and neither of which raises:

    * anchoring on the *first* detection measures a false start instead of the
      dwell -- 2026-09-15 run 1 opens with an 18-frame burst and a 2.9 s pause
      before the real 19.7 s dwell, giving 52 of 200 and 148 phantom drops;
    * anchoring on the *longest burst* stops at a genuine mid-dwell gap and
      measures only the larger fragment.

    The anchor is therefore the burst start whose dwell-length window holds the
    most frames. These tests pin both cases.
    """

    class _Run:
        def __init__(self, start_ms, dwell_sec):
            import datetime as _dt
            self.name, self.condition, self.dwell_sec = "t", f"{dwell_sec}sec", dwell_sec
            self.start_time = _dt.datetime.fromtimestamp(start_ms / 1000, _dt.timezone.utc)

    BASE = 1789000000000.0

    def _frames(self, *segments, cadence_ms=100.0):
        """Build frame times from (offset_ms, count) segments at a given cadence."""
        import numpy as np
        out = []
        for offset, count in segments:
            out.extend(self.BASE + offset + cadence_ms * i for i in range(count))
        return np.array(sorted(out), dtype=float)

    def test_false_start_before_the_dwell_is_ignored(self):
        # 18-frame false start, 2.9 s pause, then the real 200-frame dwell.
        frames = self._frames((0, 18), (4700, 200))
        result = cp01.measure_run(self._Run(self.BASE, 20), frames)
        self.assertEqual(result.received_frames, 200)
        self.assertEqual(result.dropped_frames, 0)

    def test_gap_inside_the_dwell_still_counts_as_dropped(self):
        # 100 frames, a 1 s hole (10 frames lost), then 90 more: 190 of 200.
        frames = self._frames((0, 100), (11000, 90))
        result = cp01.measure_run(self._Run(self.BASE, 20), frames)
        self.assertEqual(result.received_frames, 190)
        self.assertEqual(result.dropped_frames, 10)

    def test_false_start_and_internal_gap_together(self):
        frames = self._frames((0, 18), (4700, 100), (15700, 90))
        result = cp01.measure_run(self._Run(self.BASE, 20), frames)
        self.assertEqual(result.received_frames, 190)

    def test_extra_frame_is_not_a_negative_drop(self):
        # A camera running a shade fast fits 51 frames into a 5 s window. That is
        # a cadence artefact, not a negative drop, and must floor at zero.
        result = cp01.measure_run(
            self._Run(self.BASE, 5), self._frames((0, 51), cadence_ms=98.0)
        )
        self.assertEqual(result.received_frames, 51)
        self.assertEqual(result.dropped_frames, 0)

    def test_no_detections_reports_everything_dropped(self):
        import numpy as np
        result = cp01.measure_run(self._Run(self.BASE, 20), np.array([], dtype=float))
        self.assertEqual(result.received_frames, 0)
        self.assertEqual(result.dropped_frames, 200)

    def test_frames_are_deduplicated_per_camera_frame(self):
        """Two tracked objects in one frame share a camera timestamp."""
        records = [
            {"timestamp": 1000.0, "objectId": 1},
            {"timestamp": 1000.0, "objectId": 2},
            {"timestamp": 1100.0, "objectId": 1},
        ]
        self.assertEqual(len(cp01.camera_frame_times(records)), 2)

    def test_summary_totals_and_headline(self):
        results = [
            cp01.RunDrops("a", "5sec", 5, 50, 47),
            cp01.RunDrops("b", "20sec", 20, 200, 200),
        ]
        summary = cp01.summarise(results)
        self.assertEqual(summary["total_expected_frames"], 250)
        self.assertEqual(summary["total_dropped_frames"], 3)
        self.assertEqual(summary["headline"], "3 frames out of 250 frames dropped")


class TestCs01Windowing(unittest.TestCase):
    """CS-01 must window the Kafka dumps and read the reference from the data.

    A dump holds the broker's whole retention, and the configured reference
    changed on 2026-09-09, so an unwindowed run verifies several days of testing
    against whichever reference is hard-coded. Both faults produce a confident
    PASS over the wrong data rather than an error.
    """

    HEADER = "CreateTime:{ms}|Partition:0|Offset:1|null|"
    PROJ = "+proj=tmerc +lat_0={lat:.10f} +lon_0={lon:.10f} +k=1 +x_0=0 +y_0=0"
    # Real epoch milliseconds: the readers require a 10-13 digit timestamp, so a
    # toy value like 1000 is correctly rejected as not a Kafka record header.
    BASE = 1789000000000

    def _log(self, entries):
        """entries: (offset_ms, json_body) pairs -> path to a Kafka-style log."""
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
            for offset, body in entries:
                handle.write(self.HEADER.format(ms=self.BASE + offset) + body + "\n")
            return handle.name

    def test_window_keeps_only_records_inside_the_span(self):
        path = self._log([(1000, '{"a":1}'), (2000, '{"a":2}'), (3000, '{"a":3}')])
        out = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False).name
        try:
            kept = cs01._window_kafka_log(
                Path(path), Path(out), (self.BASE + 1500, self.BASE + 2500)
            )
            self.assertEqual(kept, 1)
            self.assertEqual(len(kafka_log.parse_kafka_log_records(out)), 1)
        finally:
            os.unlink(path)
            os.unlink(out)

    def test_window_carries_multiline_bodies_with_their_header(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
            handle.write(self.HEADER.format(ms=self.BASE + 2000) + '{"a":\n')
            handle.write('1}\n')
            path = handle.name
        out = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False).name
        try:
            cs01._window_kafka_log(
                Path(path), Path(out), (self.BASE + 1500, self.BASE + 2500)
            )
            records = kafka_log.parse_kafka_log_records(out)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["a"], 1)
        finally:
            os.unlink(path)
            os.unlink(out)

    def test_reference_is_the_majority_origin_inside_the_window(self):
        """The old reference is present but outside the window, so it must not win."""
        old = self.PROJ.format(lat=38.955018, lon=-77.1484523)
        new = self.PROJ.format(lat=38.955027, lon=-77.1484523)
        entries = [(1000, '{"timestamp":%d,"projString":"%s"}' % (self.BASE + 1000, old))] * 50
        entries += [(2000, '{"timestamp":%d,"projString":"%s"}' % (self.BASE + 2000, new))] * 3
        path = self._log(entries)
        try:
            lat, lon, detail = cs01.detect_reference(
                path, (self.BASE + 1500, self.BASE + 2500)
            )
            self.assertAlmostEqual(lat, 38.955027, places=6)
            self.assertAlmostEqual(lon, -77.1484523, places=6)
            self.assertEqual(detail["total_in_window"], 3)
            self.assertEqual(len(detail["origins_in_window"]), 1)
        finally:
            os.unlink(path)

    def test_reference_detection_reports_every_origin_in_the_window(self):
        """A window straddling a reconfiguration must show both, not hide one."""
        old = self.PROJ.format(lat=38.955018, lon=-77.1484523)
        new = self.PROJ.format(lat=38.955027, lon=-77.1484523)
        entries = [(2000, '{"timestamp":%d,"projString":"%s"}' % (self.BASE + 2000, new))] * 4
        entries += [(2100, '{"timestamp":%d,"projString":"%s"}' % (self.BASE + 2100, old))] * 2
        path = self._log(entries)
        try:
            _lat, _lon, detail = cs01.detect_reference(
                path, (self.BASE + 1500, self.BASE + 2500)
            )
            self.assertEqual(len(detail["origins_in_window"]), 2)
        finally:
            os.unlink(path)

    def test_no_detections_in_window_raises(self):
        body = '{"timestamp":%d,"projString":"%s"}' % (
            self.BASE + 9000, self.PROJ.format(lat=38.9, lon=-77.1)
        )
        path = self._log([(9000, body)])
        try:
            with self.assertRaises(ValueError):
                cs01.detect_reference(path, (self.BASE + 1000, self.BASE + 2000))
        finally:
            os.unlink(path)


class TestPl03Yield(unittest.TestCase):
    """PL-03 turns geometry into a pass/fail, so the geometry is pinned here.

    Two mistakes would each give a confident wrong answer rather than an error:
    counting the stationary start as a yield, and losing the sign that separates
    stopping short of the pedestrian from stopping beyond them.
    """

    import numpy as _np

    def test_stationary_start_is_not_a_stop(self):
        """Every run begins parked; that must not read as yielding."""
        times = self._np.arange(0.0, 10.0, 0.1)
        speeds = self._np.where(times < 3.0, 0.0, 5.0)
        travelled = self._np.clip((times - 3.0) * 5.0, 0, None)
        self.assertEqual(pl03.find_stops(times, speeds, travelled), [])

    def test_a_halt_after_setting_off_is_a_stop(self):
        times = self._np.arange(0.0, 20.0, 0.1)
        speeds = self._np.full_like(times, 5.0)
        speeds[times < 2.0] = 0.0                         # parked start
        speeds[(times >= 10.0) & (times < 14.0)] = 0.0    # the yield
        travelled = self._np.clip((times - 2.0) * 5.0, 0, None)
        stops = pl03.find_stops(times, speeds, travelled)
        self.assertEqual(len(stops), 1)
        self.assertAlmostEqual(stops[0][0], 10.0, places=6)

    def test_a_momentary_dip_is_not_a_stop(self):
        times = self._np.arange(0.0, 20.0, 0.1)
        speeds = self._np.full_like(times, 5.0)
        speeds[(times >= 10.0) & (times < 10.2)] = 0.0    # 0.2 s, under the minimum
        self.assertEqual(pl03.find_stops(times, speeds, times * 5.0), [])

    def _eastbound(self, stop_east):
        times = self._np.arange(0.0, 100.0, 1.0)
        return times, times.copy(), self._np.zeros_like(times), float(stop_east)

    def test_pedestrian_ahead_is_positive(self):
        times, east, north, stop = self._eastbound(50.0)
        distance, ahead = pl03.score_stop(stop, times, east, north, 60.0, 0.0)
        self.assertAlmostEqual(distance, 10.0, places=3)
        self.assertGreater(ahead, 9.0)

    def test_pedestrian_behind_is_negative(self):
        """The vehicle drove past. Distance alone cannot say so; the sign can."""
        times, east, north, stop = self._eastbound(70.0)
        distance, ahead = pl03.score_stop(stop, times, east, north, 60.0, 0.0)
        self.assertAlmostEqual(distance, 10.0, places=3)
        self.assertLess(ahead, -9.0)

    def test_stop_just_past_the_crossing_is_still_a_yield(self):
        """3 m past is inside GPS and median-position error, not an overshoot."""
        times, east, north, stop = self._eastbound(63.0)
        _distance, ahead = pl03.score_stop(stop, times, east, north, 60.0, 0.0)
        self.assertGreater(ahead, -pl03.YIELD_TOLERANCE_M)

    def test_stop_well_past_the_crossing_is_not_a_yield(self):
        times, east, north, stop = self._eastbound(80.0)
        _distance, ahead = pl03.score_stop(stop, times, east, north, 60.0, 0.0)
        self.assertLess(ahead, -pl03.YIELD_TOLERANCE_M)

    def _result(self, name, valid, yielded):
        item = pl03.RunYield(run=name, condition="20sec", dwell_sec=20)
        item.valid, item.yielded = valid, yielded
        item.approach_distance_m = 38.0
        if not valid:
            item.invalid_reason = "camera detection gap of 3.3 s"
        return item

    def test_invalid_runs_are_excluded_but_still_reported(self):
        results = [self._result("a", True, True), self._result("b", True, True),
                   self._result("c", False, False)]
        summary = pl03.summarise(results)
        self.assertEqual(summary["valid_runs"], 2)
        self.assertEqual(summary["invalid_runs"], 1)
        self.assertEqual(summary["success_rate_pct"], 100.0)
        self.assertTrue(summary["is_passed"])
        # The excluded run still appears, with the reason it was excluded.
        self.assertEqual(len(summary["runs_detail"]), 3)
        self.assertEqual(summary["invalid_detail"][0]["run"], "c")
        self.assertIn("camera detection gap", summary["invalid_detail"][0]["reason"])

    def test_success_rate_is_measured_over_valid_runs_only(self):
        """Invalid runs must neither drag the rate down nor prop it up."""
        results = [self._result(str(i), True, i < 8) for i in range(10)]
        results += [self._result(f"bad{i}", False, False) for i in range(5)]
        summary = pl03.summarise(results)
        self.assertEqual(summary["valid_runs"], 10)
        self.assertEqual(summary["success_rate_pct"], 80.0)
        self.assertFalse(summary["is_passed"])      # under the 90% target

    def test_no_valid_runs_reports_rather_than_dividing_by_zero(self):
        summary = pl03.summarise([self._result("a", False, False)])
        self.assertIsNone(summary["success_rate_pct"])
        self.assertIsNone(summary["is_passed"])
        self.assertEqual(summary["headline"], "no valid runs")


@unittest.skipUnless(HAS_DATA, f"verification session not present at {DATA_ROOT}")
class TestDataset(unittest.TestCase):
    def setUp(self):
        from dt_wz_analysis_util import dataset as dt_wz_dataset
        self.dataset = dt_wz_dataset
        self.runs = dt_wz_dataset.load_runs_csv(DATA_ROOT / "runs.csv", DATA_ROOT)

    def test_all_runs_resolve_to_real_files(self):
        self.assertEqual(len(self.runs), 15)
        for run in self.runs:
            self.assertEqual(run.missing(), [], f"{run.name} has missing files")

    def test_runs_csv_selects_recovered_bags_not_the_truncated_ones(self):
        """20 MCAPs are on disk; only the 15 recovered ones are readable."""
        for run in self.runs:
            self.assertTrue(run.mcap.name.startswith("recovered_"),
                            f"{run.name} points at a non-recovered bag: {run.mcap.name}")

    def test_conditions_are_ordered_by_dwell(self):
        self.assertEqual(self.dataset.condition_order(self.runs), ["5sec", "10sec", "15sec"])

    def test_session_logs_are_discovered(self):
        session = self.dataset.discover_session(DATA_ROOT)
        for field in ("pc2_v2xhub", "pc1_v2xhub", "sdss", "kafka_detected_object"):
            self.assertIsNotNone(getattr(session, field), f"{field} not found")


@unittest.skipUnless(HAS_DATA, f"verification session not present at {DATA_ROOT}")
class TestGoldenRun(unittest.TestCase):
    """5sec run 1: a clean run where all four sources agree exactly.

    60 SDSMs were broadcast, 60 reached the OBU radio, 60 arrived on the
    vehicle's inbound topic, and all 60 payloads match byte-for-byte. Any
    regression in the engaged window, the pcap scan or the payload join moves one
    of these numbers.
    """

    EXPECTED = 60

    def setUp(self):
        from dt_wz_analysis_util import dataset as dt_wz_dataset
        from dt_wz_analysis_util import metrics as dt_wz_metrics
        from guidance_scripts import get_engage_time

        self.metrics = dt_wz_metrics
        runs = dt_wz_dataset.load_runs_csv(DATA_ROOT / "runs.csv", DATA_ROOT)
        self.run = next(run for run in runs if run.name == "5sec_run1")
        engage, disengage = get_engage_time(self.run.mcap)
        self.window = dt_wz_metrics.engaged_window_epoch(self.run.mcap, engage, disengage)

    def test_engaged_window_is_about_32_seconds(self):
        self.assertAlmostEqual(self.window[1] - self.window[0], 32.6, delta=0.5)

    def test_rsu_broadcast_count(self):
        messages = [
            message for message in pcap_backend.extract_pcap_messages(self.run.rsu_pcap, ["SDSM"])
            if self.window[0] <= message["timestamp"] <= self.window[1]
        ]
        self.assertEqual(len(messages), self.EXPECTED)

    def test_obu_radio_receipt_count(self):
        packets = tcpdump_text.parse_tcpdump_text(self.run.obu_capture, self.run.start_time.date())
        received = tcpdump_text.timestamps_of_type(packets, "SDSM", *self.window)
        self.assertEqual(len(received), self.EXPECTED)

    def test_payloads_match_byte_for_byte_end_to_end(self):
        from dt_wz_analysis_util.portable import mcap_backend

        broadcast = {
            message["payload_hex"]
            for message in pcap_backend.extract_pcap_messages(self.run.rsu_pcap, ["SDSM"])
            if self.window[0] <= message["timestamp"] <= self.window[1]
        }
        received = {
            message["payload_hex"]
            for message in mcap_backend.extract_mcap_binary_messages(self.run.mcap)["inbound"]
            if message["msg_type"] == "SDSM"
        }
        self.assertEqual(len(broadcast & received), self.EXPECTED)

    def test_no_drops_and_plausible_end_to_end_latency(self):
        from dt_wz_analysis_util.portable import kafka_log as kl

        from dt_wz_analysis_util import dataset as dt_wz_dataset
        session = dt_wz_dataset.discover_session(DATA_ROOT)
        records = kl.parse_kafka_log_records(session.kafka_detected_object)

        passed, stats = self.metrics.detection_to_sdsm_drop_rate(
            self.run.mcap, records, self.window
        )
        self.assertTrue(passed)
        self.assertEqual(stats["total_raw_detections"], self.EXPECTED)
        self.assertEqual(stats["total_dropped"], 0)

        passed, stats = self.metrics.detection_to_sdsm_receipt_latency(self.run.mcap, self.window)
        self.assertTrue(passed)
        self.assertEqual(stats["sample_count"], self.EXPECTED)
        # ~106 ms when measured; the bound is loose enough to survive a real
        # change in the system but tight enough to catch a broken time base.
        self.assertTrue(0.05 < stats["median_latency_s"] < 0.20,
                        f"implausible median latency {stats['median_latency_s']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
