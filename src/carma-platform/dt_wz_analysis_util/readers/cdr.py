"""A CDR deserialiser driven by the ros2msg schema text embedded in the MCAP.

The rosbag stores every message definition it used, so there is no need for a
ROS installation or a generated-message package: parse the definition text into
a field tree, then walk the CDR buffer against it.

Scope: XCDR1 (``cdr`` message encoding, which is what rosbag2 writes) -- a
4-byte encapsulation header, then members laid out in declaration order, each
primitive aligned to its own size *relative to the start of the body*, strings
as a uint32 length (including the NUL) followed by the bytes and the NUL, and
sequences as a uint32 count followed by the elements. Bounded sequences
(``[<=N]``) are wire-identical to unbounded ones; fixed arrays (``[N]``) carry
no count.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field as dc_field

_BLOCK_SEP = "=" * 80
_MSG_PREFIX = "MSG: "

# name -> (struct code, size)
PRIMITIVES: dict[str, tuple[str, int]] = {
    "bool": ("?", 1),
    "byte": ("B", 1),
    "char": ("b", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "int64": ("q", 8),
    "uint64": ("Q", 8),
    "float32": ("f", 4),
    "float64": ("d", 8),
    "string": ("s", 0),
    "wstring": ("s", 0),
}


@dataclass(frozen=True)
class Field:
    name: str
    type_name: str
    is_array: bool = False
    array_len: int | None = None  # None => unbounded/bounded sequence
    is_primitive: bool = False


@dataclass
class MessageDef:
    name: str
    fields: list[Field] = dc_field(default_factory=list)


class SchemaError(ValueError):
    pass


def _strip_comment(line: str) -> str:
    # No string-literal defaults appear in these definitions, so a plain split
    # on '#' is safe.
    return line.split("#", 1)[0].rstrip()


def parse_schema(text: str, root_name: str) -> dict[str, MessageDef]:
    """Parse concatenated ros2msg definitions into ``{name: MessageDef}``."""
    defs: dict[str, MessageDef] = {}
    for block in text.split(_BLOCK_SEP):
        block = block.strip("\n")
        if not block.strip():
            continue
        lines = block.split("\n")
        name = root_name
        body = lines
        for i, raw in enumerate(lines):
            if raw.strip().startswith(_MSG_PREFIX):
                name = raw.strip()[len(_MSG_PREFIX) :].strip()
                body = lines[i + 1 :]
                break
        md = MessageDef(name=name)
        for raw in body:
            line = _strip_comment(raw).strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            type_tok, rest = parts[0], parts[1]
            # Constants (`uint8 FOO=1`) carry no wire bytes.
            if "=" in rest or (len(parts) > 2 and parts[2].startswith("=")):
                continue
            md.fields.append(_make_field(type_tok, rest))
        defs[name] = md
    return defs


def _make_field(type_tok: str, name: str) -> Field:
    is_array = False
    array_len: int | None = None
    base = type_tok
    if base.endswith("]"):
        base, _, spec = base[:-1].partition("[")
        is_array = True
        spec = spec.strip()
        if spec and not spec.startswith("<="):
            try:
                array_len = int(spec)
            except ValueError:
                array_len = None
    base = base.split("/")[-1] if base.count("/") == 0 else base
    is_primitive = _primitive_name(type_tok) is not None
    return Field(
        name=name,
        type_name=base,
        is_array=is_array,
        array_len=array_len,
        is_primitive=is_primitive,
    )


def _primitive_name(type_tok: str) -> str | None:
    base = type_tok.split("[")[0]
    return base if base in PRIMITIVES else None


class Decoder:
    """Decodes one message type; reusable across messages of that type."""

    def __init__(self, schema_text: str, root_name: str):
        self.defs = parse_schema(schema_text, root_name)
        self.root = root_name
        # Definitions reference types both fully-qualified
        # (``geometry_msgs/Pose``) and with the ``/msg/`` infix stripped, so
        # index every definition by its last path segment too.
        self._by_suffix: dict[str, MessageDef] = {}
        for name, md in self.defs.items():
            self._by_suffix.setdefault(name.split("/")[-1], md)
            self._by_suffix.setdefault(name, md)

    def _lookup(self, type_name: str) -> MessageDef:
        md = self._by_suffix.get(type_name) or self._by_suffix.get(type_name.split("/")[-1])
        if md is None:
            raise SchemaError(f"no definition for type {type_name!r} in {self.root}")
        return md

    # -- buffer primitives ---------------------------------------------------
    def _align(self, pos: int, size: int) -> int:
        if size <= 1:
            return pos
        # Alignment is measured from the end of the 4-byte encapsulation header.
        rem = (pos - 4) % size
        return pos if rem == 0 else pos + (size - rem)

    def decode(self, data: bytes) -> dict:
        if len(data) < 4:
            raise SchemaError("buffer shorter than CDR encapsulation header")
        self._little = data[1] in (1, 3)
        value, _ = self._read_struct(self._lookup(self.root), data, 4)
        return value

    def _endian(self) -> str:
        return "<" if self._little else ">"

    def _read_struct(self, md: MessageDef, data: bytes, pos: int) -> tuple[dict, int]:
        out: dict = {}
        for f in md.fields:
            out[f.name], pos = self._read_field(f, data, pos)
        return out, pos

    def _read_field(self, f: Field, data: bytes, pos: int):
        if f.is_array:
            if f.array_len is None:
                pos = self._align(pos, 4)
                (count,) = struct.unpack_from(self._endian() + "I", data, pos)
                pos += 4
            else:
                count = f.array_len
            if f.is_primitive and f.type_name in ("uint8", "byte", "char", "int8"):
                # Fast path: byte blobs are the bulk of inbound_binary_msg.
                code, size = PRIMITIVES[f.type_name]
                raw = bytes(data[pos : pos + count])
                return raw, pos + count * size
            items = []
            for _ in range(count):
                value, pos = self._read_scalar(f, data, pos)
                items.append(value)
            return items, pos
        return self._read_scalar(f, data, pos)

    def _read_scalar(self, f: Field, data: bytes, pos: int):
        if f.is_primitive:
            name = f.type_name
            if name in ("string", "wstring"):
                pos = self._align(pos, 4)
                (length,) = struct.unpack_from(self._endian() + "I", data, pos)
                pos += 4
                raw = data[pos : pos + max(length - 1, 0)]
                pos += length
                return raw.decode("utf-8", "replace"), pos
            code, size = PRIMITIVES[name]
            pos = self._align(pos, size)
            (value,) = struct.unpack_from(self._endian() + code, data, pos)
            return value, pos + size
        return self._read_struct(self._lookup(f.type_name), data, pos)


def make_decoders(summary, topics: set[str]) -> dict[int, tuple[str, Decoder]]:
    """Build ``{channel_id: (topic, Decoder)}`` for the requested topics."""
    out: dict[int, tuple[str, Decoder]] = {}
    for cid, ch in summary.channels.items():
        if ch.topic not in topics:
            continue
        if ch.message_encoding != "cdr":
            raise SchemaError(f"{ch.topic}: unexpected encoding {ch.message_encoding!r}")
        sc = summary.schemas[ch.schema_id]
        out[cid] = (ch.topic, Decoder(sc.data.decode(), sc.name))
    return out
