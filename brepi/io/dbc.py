"""Decoder for DATASUS ``.dbc`` (and bare ``.dbf``) payloads.

``.dbc`` is a dBase III table whose *data* block is compressed with a
PKWare-DCL-derived scheme; the 32-byte file header and the field descriptor
array are stored uncompressed at the front, which is why
:func:`dbf_header` can audit schema drift without paying for decompression of
the rows. Decompression is delegated to ``datasus_dbc``.

Three traps are handled here explicitly.

*Mixed extensions.* The public tree contains files named ``.dbc`` that are in
fact uncompressed ``.dbf``. Rather than trusting the name, the header is read
and the declared geometry (``header_len + n_records * record_len + 1``) is
compared against the file size; an exact match means the table is already
plain, and decompression is skipped.

*Type coercion.* Every column is returned as :class:`polars.Utf8`. DATASUS
categorical codes are zero-padded fixed-width strings — IBGE municipality
codes (``350000``/``355030``), CID-10 codes (``A279``), UF codes (``11``) —
and any numeric coercion destroys the leading zeros that make them joinable.
The values are taken from the raw field bytes, not from ``dbfread``'s parsed
Python objects, so what lands in the frame is exactly what is on disk.
``D``/``T`` fields are the one exception: those are parsed and re-emitted as
ISO-8601 strings.

*Encoding.* The tree is ``iso-8859-1``, never UTF-8, and a minority of records
carry bytes that are invalid even there; decode errors are replaced rather
than raised, because one corrupt accented municipality name must not abort a
20-year extraction.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import datasus_dbc
import polars as pl
from dbfread import DBF, FieldParser

from brepi.config import DATASUS_ENCODING

__all__ = [
    "read_dbc",
    "read_dbc_bytes",
    "read_dbc_where",
    "dbf_header",
    "field_offsets",
    "is_plain_dbf",
    "DbcDecodeError",
]


class DbcDecodeError(RuntimeError):
    """A payload could not be decoded as either a ``.dbc`` or a ``.dbf``.

    Deterministic: raised immediately, never retried. A truncated download
    surfaces here rather than as a silently short table.
    """


# --------------------------------------------------------------------------
# Header inspection
# --------------------------------------------------------------------------

_HEADER_STRUCT = "<BBBBLHH"  # version, yy, mm, dd, n_records, header_len, record_len
_HEADER_SIZE = struct.calcsize(_HEADER_STRUCT)
_FIELD_DESC_SIZE = 32


def _read_geometry(path: Path) -> tuple[int, int, int, int]:
    """Return ``(version, n_records, header_len, record_len)`` from the header."""
    with path.open("rb") as fh:
        raw = fh.read(_HEADER_SIZE)
    if len(raw) < _HEADER_SIZE:
        raise DbcDecodeError(f"{path} is shorter than a dBase header ({len(raw)} bytes)")
    version, _yy, _mm, _dd, n_records, header_len, record_len = struct.unpack(
        _HEADER_STRUCT, raw
    )
    return version, n_records, header_len, record_len


def is_plain_dbf(path: Path) -> bool:
    """True when ``path`` is an uncompressed dBase table despite its name.

    The test is geometric, not extension-based: a plain table's size equals
    ``header_len + n_records * record_len`` plus at most a few trailing bytes
    (the ``0x1A`` EOF marker, which some writers omit). A ``.dbc`` carries the
    same declared geometry but far fewer actual bytes, so the comparison
    separates the two without attempting a decode.
    """
    try:
        _version, n_records, header_len, record_len = _read_geometry(path)
    except DbcDecodeError:
        return False
    if header_len <= 0 or record_len <= 0:
        return False
    expected = header_len + n_records * record_len
    actual = path.stat().st_size
    return expected <= actual <= expected + 8


def _header_terminator_ok(path: Path) -> bool:
    """True when the field-descriptor array ends in the byte dBase mandates.

    CNES ``ST`` tables (and a scattering of others) terminate the descriptor
    array with ``0x00`` instead of ``0x0D``. ``dbfread`` loops on the
    terminator, so it walks straight past the array into the first data record
    and dies with a struct-unpack error many fields later. The malformation is
    repaired on the temporary copy, never on the cached original.
    """
    _version, _n, header_len, _rec = _read_geometry(path)
    if header_len < 33:
        return False
    with path.open("rb") as fh:
        fh.seek(header_len - 1)
        return fh.read(1) in (b"\r", b"\n")


def _repair_header_terminator(path: Path) -> None:
    _version, _n, header_len, _rec = _read_geometry(path)
    with path.open("r+b") as fh:
        fh.seek(header_len - 1)
        fh.write(b"\r")


def _decompress_to(path: Path, dest_dir: Path) -> Path:
    """Materialise ``path`` as a plain, readable ``.dbf`` inside ``dest_dir``."""
    dbf_path = dest_dir / (path.stem + ".dbf")
    if is_plain_dbf(path):
        shutil.copyfile(path, dbf_path)
        if not _header_terminator_ok(dbf_path):
            _repair_header_terminator(dbf_path)
        return dbf_path
    try:
        datasus_dbc.decompress(str(path), str(dbf_path))
    except Exception as exc:  # deterministic: bad bytes, not a transient fault
        raise DbcDecodeError(f"could not decompress {path.name}: {exc}") from exc
    if not dbf_path.exists() or dbf_path.stat().st_size == 0:
        raise DbcDecodeError(f"decompression of {path.name} produced no output")
    if not _header_terminator_ok(dbf_path):
        _repair_header_terminator(dbf_path)
    return dbf_path


def dbf_header(path: Path) -> list[dict[str, Any]]:
    """Describe a table's fields without reading a single row.

    Returns one record per field with ``name``, ``type`` (the one-character
    dBase type code), ``length`` and ``decimals``, in physical order. This is
    the input to schema-drift auditing: SINAN adds, drops and *resizes* fields
    between years, and a resized field is a silent truncation risk that a
    name-only comparison misses.

    The descriptor array is uncompressed even in a ``.dbc``, so this is read
    directly from the front of the file.
    """
    path = Path(path)
    _version, _n_records, header_len, _record_len = _read_geometry(path)
    n_fields = max((header_len - 33) // _FIELD_DESC_SIZE, 0)
    with path.open("rb") as fh:
        fh.seek(32)
        blob = fh.read(n_fields * _FIELD_DESC_SIZE)
    out: list[dict[str, Any]] = []
    for i in range(n_fields):
        desc = blob[i * _FIELD_DESC_SIZE : (i + 1) * _FIELD_DESC_SIZE]
        if len(desc) < _FIELD_DESC_SIZE or desc[0] in (0x0D, 0x00):
            break
        name = desc[:11].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
        if not name:
            break
        out.append(
            {
                "name": name,
                "type": chr(desc[11]),
                "length": desc[16],
                "decimals": desc[17],
            }
        )
    if not out:
        raise DbcDecodeError(f"no field descriptors found in {path.name}")
    return out


# --------------------------------------------------------------------------
# Row decoding
# --------------------------------------------------------------------------


def _clean(data: bytes) -> str | None:
    text = data.decode(DATASUS_ENCODING, errors="replace").replace("\x00", " ").strip()
    return text or None


class _RawStringParser(FieldParser):
    """Yield every field as text taken from its on-disk bytes.

    Numeric (``N``/``F``/``I``/``B``/``Y``) fields are deliberately *not*
    parsed: in DATASUS extracts they hold zero-padded categorical codes far
    more often than they hold quantities, and ``dbfread``'s int/float parse is
    lossy for those. Callers that want a number cast the column explicitly and
    thereby say so.
    """

    def parse(self, field: Any, data: bytes) -> str | None:  # type: ignore[override]
        kind = field.type
        if kind in ("D", "T", "@"):
            try:
                value = super().parse(field, data)
            except Exception:
                return _clean(data)
            if isinstance(value, (_dt.datetime, _dt.date)):
                return value.isoformat()
            return _clean(data)
        if kind == "L":
            try:
                value = super().parse(field, data)
            except Exception:
                return _clean(data)
            if value is None:
                return None
            return "T" if value else "F"
        return _clean(data)


def _iter_records(dbf: DBF) -> Iterator[dict[str, Any]]:
    for record in dbf:
        yield dict(record)


def _dbf_to_polars(path: Path) -> pl.DataFrame:
    table = DBF(
        str(path),
        encoding=DATASUS_ENCODING,
        char_decode_errors="replace",
        parserclass=_RawStringParser,
        ignore_missing_memofile=True,
        load=False,
    )
    names = [str(n).strip() for n in table.field_names]
    columns: dict[str, list[str | None]] = {name: [] for name in names}
    n = 0
    for record in _iter_records(table):
        for raw_name, name in zip(table.field_names, names):
            columns[name].append(record.get(raw_name))
        n += 1
    if not columns:
        raise DbcDecodeError(f"{path.name} declares no fields")
    return pl.DataFrame(
        columns,
        schema={name: pl.Utf8 for name in names},
        strict=False,
    )


def read_dbc(path: Path) -> pl.DataFrame:
    """Decode one ``.dbc``/``.dbf`` file into an all-``Utf8`` polars frame.

    Column order follows the physical field order. Values are whitespace- and
    NUL-stripped; empty fields become null rather than ``""``, so that
    ``is_null`` means "not recorded" consistently across sources.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if is_plain_dbf(path) and _header_terminator_ok(path):
        return _dbf_to_polars(path)
    with tempfile.TemporaryDirectory(prefix="brepi_dbc_") as tmp:
        return _dbf_to_polars(_decompress_to(path, Path(tmp)))


# --------------------------------------------------------------------------
# Byte-level predicate pushdown
# --------------------------------------------------------------------------
#
# ``dbfread`` walks every record in Python and instantiates one dict per row.
# That is acceptable for a 0.3 MB SIM state-year but not for SIH RD, where one
# UF-month is ~10^5-10^6 rows across ~110 fields and a national 17-year
# extraction is ~6,000 such files. Since a dBase data block is a fixed-width
# byte matrix, the selection predicate can be evaluated on the raw bytes with
# numpy before any Python object exists: a 4-byte column slice, an ``S4`` view
# and ``np.char.startswith`` reduce a million rows to the handful that mention
# A27, and only those rows are ever materialised as strings.
#
# This is exact, not approximate: the same bytes ``dbfread`` would have decoded
# are compared, only earlier and in C.


def field_offsets(path: Path) -> list[dict[str, Any]]:
    """Field descriptors annotated with their byte offset inside a record.

    Offsets start at 1 because byte 0 of every dBase record is the deletion
    flag (``0x20`` live, ``0x2A`` deleted).
    """
    fields = dbf_header(Path(path))
    offset = 1
    for f in fields:
        f["offset"] = offset
        offset += int(f["length"])
    return fields


def _decode_column(block: "Any", offset: int, length: int) -> list[str | None]:
    import numpy as np

    raw = np.ascontiguousarray(block[:, offset : offset + length]).tobytes()
    view = np.frombuffer(raw, dtype=f"S{length}")
    return [_clean(v) for v in view.tolist()]


def read_dbc_where(
    path: Path,
    *,
    any_prefix: dict[str, Sequence[str]],
    fields: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Decode only the rows whose value in some column has a wanted prefix.

    ``any_prefix`` maps column name -> accepted prefixes; a row is kept when
    *any* listed column's stripped value starts with *any* of its prefixes
    (``{"DIAG_PRINC": ["A27"], "DIAG_SECUN": ["A27"]}``). Columns named in
    ``any_prefix`` but absent from the layout are ignored; if none of them are
    present the call raises, because a silently-empty filter would look like a
    genuine absence of disease.

    ``fields`` projects the surviving rows (default: every field). All values
    are :class:`polars.Utf8`, matching :func:`read_dbc`.
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix="brepi_dbc_") as tmp:
        if is_plain_dbf(path):
            plain = path
        else:
            plain = _decompress_to(path, Path(tmp))
        _version, n_records, header_len, record_len = _read_geometry(plain)
        descs = field_offsets(plain)
        by_name = {d["name"]: d for d in descs}

        wanted = {c: list(p) for c, p in any_prefix.items() if c in by_name}
        if not wanted:
            raise DbcDecodeError(
                f"{path.name}: none of {list(any_prefix)} present; "
                f"columns are {[d['name'] for d in descs][:20]}"
            )

        available = max((plain.stat().st_size - header_len) // record_len, 0)
        n = min(n_records, available)
        names = [d["name"] for d in descs]
        projection = [f for f in (fields or names) if f in by_name]
        if n == 0:
            return pl.DataFrame(schema={f: pl.Utf8 for f in projection})

        block = np.fromfile(
            plain, dtype=np.uint8, count=n * record_len, offset=header_len
        ).reshape(n, record_len)

        mask = block[:, 0] == 0x20  # live records only
        hit = np.zeros(n, dtype=bool)
        for col, prefixes in wanted.items():
            d = by_name[col]
            length = int(d["length"])
            raw = np.ascontiguousarray(
                block[:, d["offset"] : d["offset"] + length]
            ).tobytes()
            view = np.char.strip(np.frombuffer(raw, dtype=f"S{length}"))
            for prefix in prefixes:
                hit |= np.char.startswith(view, prefix.strip().upper().encode("ascii"))
        keep = mask & hit
        if not keep.any():
            return pl.DataFrame(schema={f: pl.Utf8 for f in projection})

        rows = np.ascontiguousarray(block[keep])
        del block
        data = {
            f: _decode_column(rows, by_name[f]["offset"], int(by_name[f]["length"]))
            for f in projection
        }
        return pl.DataFrame(data, schema={f: pl.Utf8 for f in projection}, strict=False)


def read_dbc_bytes(payload: bytes, *, name: str) -> pl.DataFrame:
    """Decode an in-memory ``.dbc``/``.dbf`` payload.

    ``datasus_dbc`` is file-oriented, so the bytes are spilled to a temporary
    file that is removed on the way out. ``name`` is used only for the
    temporary file name and for error messages.
    """
    if not payload:
        raise DbcDecodeError(f"empty payload for {name}")
    safe = Path(name).name or "payload.dbc"
    with tempfile.TemporaryDirectory(prefix="brepi_dbc_") as tmp:
        src = Path(tmp) / safe
        src.write_bytes(payload)
        return read_dbc(src)
