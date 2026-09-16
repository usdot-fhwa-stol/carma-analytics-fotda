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

from portable import kafka_log, pcap_backend, tcpdump_text  # noqa: E402

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


@unittest.skipUnless(HAS_DATA, f"verification session not present at {DATA_ROOT}")
class TestDataset(unittest.TestCase):
    def setUp(self):
        import dt_wz_dataset
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
        import dt_wz_dataset
        import dt_wz_metrics
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
        from portable import mcap_backend

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
        from portable import kafka_log as kl

        import dt_wz_dataset
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
