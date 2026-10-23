"""Minimal reader for R .Rds files (XDR serialization v2/v3) sufficient for data.frame/data.table
with integer, double, logical and character columns. Dates (Date/IDate) are returned as day numbers."""
import gzip
import struct
import sys

import numpy as np
import pandas as pd

NILVALUE, REFSXP, GLOBALENV, UNBOUND, MISSINGARG, BASENS, EMPTYENV, BASEENV = 254, 255, 253, 252, 251, 247, 242, 241
SPECIAL = {NILVALUE, GLOBALENV, UNBOUND, MISSINGARG, BASENS, EMPTYENV, BASEENV}


class R:
    def __init__(self, data):
        self.b, self.i, self.refs = data, 0, []

    def int(self):
        v = struct.unpack_from(">i", self.b, self.i)[0]
        self.i += 4
        return v

    def length(self):
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t in SPECIAL:
            return None
        if t == REFSXP:
            idx = flags >> 8
            if idx == 0:
                idx = self.int()
            return self.refs[idx - 1]
        if t == 1:                       # SYMSXP
            s = self.item()
            self.refs.append(s)
            return s
        if t in (2, 6, 17):              # LISTSXP / LANGSXP / DOTSXP (pairlists)
            out = {}
            while True:
                attr = self.item() if has_attr else None  # noqa: F841
                tag = self.item() if has_tag else None
                car = self.item()
                out[tag if tag is not None else len(out)] = car
                f2 = self.int()
                t2 = f2 & 0xFF
                if t2 in SPECIAL:
                    return out
                if t2 not in (2, 6, 17):
                    self.i -= 4
                    out["__cdr__"] = self.item()
                    return out
                has_attr, has_tag = bool(f2 & (1 << 9)), bool(f2 & (1 << 10))
        if t == 4:                       # ENVSXP
            self.refs.append({})
            self.int()                   # locked
            for _ in range(4):           # enclos, frame, hashtab, attrib
                self.item()
            return {}
        if t == 22:                      # EXTPTRSXP
            self.refs.append(None)
            self.item(); self.item()
            if has_attr:
                self.item()
            return None
        if t == 9:                       # CHARSXP
            n = self.int()
            if n == -1:
                return None
            s = self.b[self.i:self.i + n].decode("utf-8", "replace")
            self.i += n
            return s
        if t in (10, 13):                # LGLSXP / INTSXP
            n = self.length()
            v = np.frombuffer(self.b, dtype=">i4", count=n, offset=self.i).astype(np.int64)
            self.i += 4 * n
            val = v
        elif t == 14:                    # REALSXP
            n = self.length()
            val = np.frombuffer(self.b, dtype=">f8", count=n, offset=self.i).astype(float)
            self.i += 8 * n
        elif t == 16:                    # STRSXP
            n = self.length()
            val = [self.item() for _ in range(n)]
        elif t in (19, 20):              # VECSXP / EXPRSXP
            n = self.length()
            val = [self.item() for _ in range(n)]
        else:
            raise ValueError(f"unsupported SEXP type {t} at {self.i}")
        attrs = self.item() if has_attr else {}
        return {"__type__": t, "value": val, "attrs": attrs or {}}


def read_rds(path) -> pd.DataFrame:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    assert raw[:2] == b"X\n", "only XDR format supported"
    r = R(raw)
    r.i = 2
    version = r.int(); r.int(); r.int()
    if version == 3:
        n = r.int(); r.i += n
    obj = r.item()
    names = obj["attrs"]["names"]["value"]
    cols = {}
    for name, col in zip(names, obj["value"]):
        cls = (col["attrs"].get("class") or {}).get("value", []) if isinstance(col["attrs"].get("class"), dict) else []
        v = col["value"]
        if col["__type__"] == 13:
            v = np.where(v == -2147483648, np.nan, v)
        if col["__type__"] == 10:
            v = np.where(v == -2147483648, np.nan, v).astype(float)
        if "Date" in cls or "IDate" in cls:
            v = pd.to_datetime(np.asarray(v, dtype=float), unit="D", origin="unix")
        cols[name] = v
    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = read_rds(sys.argv[1])
    print(df.shape)
    print(df.dtypes)
    if "Inspection_Date" in df:
        print(df.Inspection_Date.min(), df.Inspection_Date.max())
    if len(sys.argv) > 2:
        df.to_csv(sys.argv[2], index=False)
        print("wrote", sys.argv[2])
