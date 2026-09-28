"""Point cloud readers and writers.

Every reader returns a ``dict`` with an ``"xyz"`` array (N, 3, float64) plus any
per-point attribute columns found in the file (float32 / int arrays of length N).
Supported inputs: LAS/LAZ, PLY (ascii + binary), PCD (ascii + binary),
CloudCompare-style ASCII (.txt/.csv/.asc/.xyz/.pts), NPY/NPZ.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

Cloud = Dict[str, np.ndarray]

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _norm_name(name: str) -> str:
    name = name.strip().lstrip("/").rstrip("\\").strip().lower()
    name = re.sub(r"[^0-9a-z]+", "_", name).strip("_")
    return {"red": "r", "green": "g", "blue": "b", "rf": "r", "gf": "g", "bf": "b",
            "scalar_intensity": "intensity"}.get(name, name)


def _finalize(columns: Dict[str, np.ndarray]) -> Cloud:
    cols = {_norm_name(k): np.asarray(v) for k, v in columns.items()}
    if not all(c in cols for c in "xyz"):
        raise ValueError(f"cloud has no x/y/z columns (found: {sorted(cols)})")
    out: Cloud = {"xyz": np.stack([cols.pop(c) for c in "xyz"], axis=1).astype(np.float64)}
    if all(c in cols for c in "rgb"):
        rgb = np.stack([cols.pop(c) for c in "rgb"], axis=1).astype(np.float32)
        out["rgb"] = rgb / (65535.0 if rgb.max() > 255 else 255.0 if rgb.max() > 1 else 1.0)
    for k, v in cols.items():
        if v.ndim == 1 and len(v) == len(out["xyz"]) and np.issubdtype(v.dtype, np.number):
            out[k] = v
    finite = np.isfinite(out["xyz"]).all(axis=1)
    if not finite.all():  # scanners export NaN/inf returns; drop them everywhere
        out = {k: v[finite] for k, v in out.items()}
    return out


# --------------------------------------------------------------------------- readers
def read_ply(path: str) -> Cloud:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, elements, cur = None, [], None
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, count = line.split()
                cur = {"name": name, "count": int(count), "props": []}
                elements.append(cur)
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] == "list":
                    cur["props"].append((parts[4], "list", parts[2], parts[3]))
                else:
                    cur["props"].append((parts[2], _PLY_TYPES[parts[1]]))
            elif line == "end_header":
                break
        vertex = elements[0]
        assert vertex["name"] == "vertex", "first PLY element must be 'vertex'"
        if any(p[1] == "list" for p in vertex["props"]):
            raise ValueError("list properties on vertices are not supported")
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=vertex["count"], ndmin=2)
            cols = {p[0]: data[:, i] for i, p in enumerate(vertex["props"])}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(p[0], endian + p[1]) for p in vertex["props"]])
            arr = np.frombuffer(f.read(dtype.itemsize * vertex["count"]), dtype=dtype)
            cols = {n: arr[n].astype(arr[n].dtype.newbyteorder("=")) for n in arr.dtype.names}
    return _finalize(cols)


def read_pcd(path: str) -> Cloud:
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == "DATA":
                break
        names, sizes, types = header["FIELDS"], header["SIZE"], header["TYPE"]
        counts = header.get("COUNT", ["1"] * len(names))
        n = int(header["POINTS"][0])
        kind = header["DATA"][0]
        if kind == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            return _finalize({nm: data[:, i] for i, nm in enumerate(names)})
        if kind != "binary":
            raise ValueError(f"PCD '{kind}' encoding is not supported; re-save as binary/ascii")
        dtype = np.dtype([(nm, f"<{t.lower()}{s}", (int(c),) if int(c) > 1 else ())
                          for nm, s, t, c in zip(names, sizes, types, counts)])
        arr = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype)
    return _finalize({nm: arr[nm] for nm in names if arr[nm].ndim == 1})


def read_las(path: str, bounds: Optional[Sequence[float]] = None,
             chunk_size: int = 5_000_000) -> Cloud:
    try:
        import laspy
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading LAS/LAZ needs `pip install 'laspy[lazrs]'`") from e
    parts: List[Cloud] = []
    with laspy.open(path) as reader:
        dims = [d for d in reader.header.point_format.dimension_names]
        keep = [d for d in dims if d.lower() in
                ("intensity", "return_number", "number_of_returns", "classification",
                 "red", "green", "blue")]
        for pts in reader.chunk_iterator(chunk_size):
            xyz = np.stack([np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)], axis=1)
            mask = _bounds_mask(xyz, bounds)
            cols = {"x": xyz[mask, 0], "y": xyz[mask, 1], "z": xyz[mask, 2]}
            for d in keep:
                cols[d] = np.asarray(pts[d])[mask]
            parts.append(_finalize(cols))
    return concat(parts)


def read_ascii(path: str) -> Cloud:
    import pandas as pd

    with open(path, "r", errors="ignore") as f:
        first = f.readline()
    sep = ";" if ";" in first else "," if "," in first else r"\s+"
    has_header = bool(re.search(r"[a-zA-Z]", first))
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None, engine="c", skipinitialspace=True)
    if not has_header:
        df.columns = ["x", "y", "z"] + [f"f{i}" for i in range(df.shape[1] - 3)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return _finalize({c: df[c].to_numpy() for c in df.columns})


def read_npy(path: str) -> Cloud:
    if path.endswith(".npz"):
        z = np.load(path)
        return _finalize({k: z[k] for k in z.files}) if "xyz" not in z.files else \
            {k: z[k] for k in z.files}
    arr = np.load(path)
    cols = {"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]}
    cols.update({f"f{i}": arr[:, i] for i in range(3, arr.shape[1])})
    return _finalize(cols)


READERS = {".ply": read_ply, ".pcd": read_pcd, ".las": read_las, ".laz": read_las,
           ".txt": read_ascii, ".csv": read_ascii, ".asc": read_ascii, ".xyz": read_ascii,
           ".pts": read_ascii, ".npy": read_npy, ".npz": read_npy}


def read_cloud(path: str, bounds: Optional[Sequence[float]] = None) -> Cloud:
    ext = os.path.splitext(path)[1].lower()
    if ext not in READERS:
        raise ValueError(f"unsupported point cloud format: {path}")
    cloud = read_las(path, bounds) if ext in (".las", ".laz") else READERS[ext](path)
    if bounds is not None and ext not in (".las", ".laz"):
        cloud = subset(cloud, _bounds_mask(cloud["xyz"], bounds))
    return cloud


def expand_paths(patterns: Iterable[str] | str) -> List[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: List[str] = []
    for p in patterns:
        hits = sorted(glob.glob(os.path.expanduser(p)))
        if not hits:
            raise FileNotFoundError(f"no files match {p!r}")
        paths.extend(hits)
    return paths


def read_clouds(patterns: Iterable[str] | str, bounds: Optional[Sequence[float]] = None) -> Cloud:
    return concat([read_cloud(p, bounds) for p in expand_paths(patterns)])


# --------------------------------------------------------------------------- helpers
def _bounds_mask(xyz: np.ndarray, bounds: Optional[Sequence[float]]) -> np.ndarray:
    if bounds is None:
        return np.ones(len(xyz), dtype=bool)
    xmin, ymin, xmax, ymax = bounds
    return (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)


def subset(cloud: Cloud, idx: np.ndarray) -> Cloud:
    return {k: v[idx] for k, v in cloud.items()}


def concat(clouds: List[Cloud]) -> Cloud:
    if len(clouds) == 1:
        return clouds[0]
    common = set.intersection(*(set(c) for c in clouds))
    return {k: np.concatenate([c[k] for c in clouds]) for k in clouds[0] if k in common}


# --------------------------------------------------------------------------- writers
def write_ply(path: str, xyz: np.ndarray, fields: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Binary PLY with double xyz (keeps UTM precision) and scalar fields CloudCompare can show."""
    fields = dict(fields or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [("x", xyz[:, 0].astype("<f8")), ("y", xyz[:, 1].astype("<f8")), ("z", xyz[:, 2].astype("<f8"))]
    rgb = fields.pop("rgb", None)
    if rgb is not None:
        rgb8 = np.clip(np.asarray(rgb) * (255 if rgb.max() <= 1 else 1), 0, 255).astype("u1")
        cols += [("red", rgb8[:, 0]), ("green", rgb8[:, 1]), ("blue", rgb8[:, 2])]
    for name, v in fields.items():
        v = np.asarray(v)
        v = v.astype("<i4") if np.issubdtype(v.dtype, np.integer) else v.astype("<f4")
        cols.append((re.sub(r"[^0-9A-Za-z_]", "_", name), v))
    ply_t = {"<f8": "double", "<f4": "float", "<i4": "int", "|u1": "uchar"}
    dtype = np.dtype([(n, v.dtype.str) for n, v in cols])
    arr = np.empty(len(xyz), dtype=dtype)
    for n, v in cols:
        arr[n] = v
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property {ply_t[v.dtype.str]} {n}" for n, v in cols] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes())


def write_las(path: str, xyz: np.ndarray, classification: np.ndarray,
              extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    for name, v in (extra or {}).items():
        las.add_extra_dim(laspy.ExtraBytesParams(name=name[:32], type=np.float32))
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = classification.astype(np.uint8)
    for name, v in (extra or {}).items():
        las[name[:32]] = v.astype(np.float32)
    las.write(path)


def write_cloud(path: str, xyz: np.ndarray, fields: Dict[str, np.ndarray], label_key: str = "label") -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        extra = {k: v for k, v in fields.items() if k not in (label_key, "rgb") and np.ndim(v) == 1}
        write_las(path, xyz, fields[label_key], extra)
    else:
        write_ply(path, xyz, fields)
