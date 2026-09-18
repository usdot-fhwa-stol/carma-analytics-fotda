# `detections.csv` columns

One row per FLIR camera detection. Every `t_*` column is an absolute epoch
timestamp in **milliseconds UTC**; `lat_*_ms` is that stage measured from the
camera detection; `d_a__b_ms` is the cost of the single hop from `a` to `b`.

| Column | Host | Meaning |
| --- | --- | --- |
| `t_flir_detect` | flir | FLIR detection timestamp (sensor-reported). |
| `t_flir_log` | pc2 | FLIRCameraDriverPlugin publishes to TMX bus. |
| `t_streets_rx` | pc2 | CARMAStreetsPlugin receives SensorDetectedObject. |
| `t_kafka_produce` | pc2 | CARMAStreetsPlugin produces to Kafka. |
| `t_kafka_sdo_create` | pc2 | Kafka broker CreateTime (detection topic). |
| `t_sdss_consume` | pc2 | sensor_data_sharing_service consumes detection. |
| `t_sdss_send` | pc2 | sensor_data_sharing_service sends SDSM. |
| `t_kafka_sdsm_create` | pc2 | Kafka broker CreateTime (SDSM topic). |
| `t_streets_sdsm_rx` | pc2 | CARMAStreetsPlugin consumes SDSM. |
| `t_streets_encode` | pc2 | CARMAStreetsPlugin UPER-encodes SDSM. |
| `t_pc2_tena_rx` | pc2 | TenaV2XPlugin reads SDSM off TMX bus. |
| `t_pc2_tena_tx` | pc2 | TenaV2XPlugin sends over TENA. |
| `t_pc1_tena_rx` | pc1 | TenaV2XPlugin observer receives from TENA. |
| `t_pc1_bus` | pc1 | SDSM published onto pc1 TMX bus. |
| `t_immediate_fwd` | pc1 | ImmediateForwardPlugin sends to RSU. |
| `t_rsu_broadcast` | rsu | RSU broadcasts SDSM over the air. **RSU transmit instant** (earlier sessions captured the OBU instead). |
| `t_obu_radio_rx` | obu | OBU radio receives SDSM (count-paired). **Paired by time, not payload, when the OBU capture carries none.** |
| `t_ros_inbound` | ros | inbound_binary_msg (recorder receive). |
| `t_ros_j3224` | ros | incoming_j3224_sdsm (recorder receive). |
| `t_ros_fused` | ros | fused_external_objects (recorder receive). |

| Column | Meaning |
| --- | --- |
| `condition` | Pedestrian dwell condition |
| `run` | Run identifier, `<condition>_run<N>` |
| `object_id` | FLIR track id; unique within a run |
| `sdsm_uper_hex` | The SDSM's ASN.1-UPER bytes, the join key from encode to vehicle |
| `stages_reached` | How many stages carry a timestamp for this detection |
| `reached_vehicle` | Whether it arrived on any vehicle-side topic |
| `last_stage_reached` | The furthest stage reached, i.e. where it was lost |
