"""
GPU-Accelerated Columnar Analytics Engine
==========================================
Architecture:
  - Data stored on SSD in columnar format (.col files per column, .meta JSON)
  - Loaded into pinned (page-locked) host RAM -> single DMA transfer to GPU
  - Generic CUDA kernels: filter, scan, scatter, groupby SUM/COUNT/MIN/MAX/AVG
  - Tkinter GUI: table browser, column selector, condition builder, result viewer
"""

import os
import json
import time
import struct
import threading
import collections
from pathlib import Path
from typing import Optional

import numpy as np

# -- GPU imports (optional: graceful fallback to CPU) --------------------------
_cuda_context = None   # global context reference for worker threads
_cuda_arch    = "sm_60"  # overridden below once device is queried

try:
    import pycuda.autoinit               # creates + pushes context on main thread
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray

    # Capture arch while context is current (do NOT pop yet: SourceModule needs it)
    _cc = cuda.Device(0).compute_capability()
    _cuda_arch = f"sm_{_cc[0]}{_cc[1]}"
    _cuda_context = cuda.Context.get_current()
    # Context stays pushed on main thread until after kernel compilation below

    GPU_AVAILABLE = True
except Exception as e:
    print(f"[INFO] GPU not available ({e}), using CPU fallback.")
    GPU_AVAILABLE = False

# -- Tkinter --------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont


# ===============================================================================
#  CONFIG
# ===============================================================================
GPU_MEM_BYTES   = 4 * 1024 * 1024 * 1024   # 4 GB (RTX 2050)
SAFETY_FACTOR   = 0.5                        # use 50 % of GPU RAM per batch
BLOCK_SIZE      = 256                        # CUDA threads per block
DATA_DIR        = Path("./columnar_data")    # root directory for column files
DATA_DIR.mkdir(exist_ok=True)


# ===============================================================================
#  CUDA KERNELS  (generic - operate on float64 values; cast on upload)
# ===============================================================================
CUDA_SOURCE = r"""
extern "C" {

/* --- 1. FILTER ------------------------------------------------------------
   Supported ops:  0 = >   1 = >=   2 = <   3 = <=   4 = ==   5 = !=
*/
__global__ void filter_kernel(
    const double *col,
    unsigned char *mask,
    double threshold,
    int op,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        double v = col[i];
        unsigned char result = 0;
        switch(op){
            case 0: result = (v >  threshold) ? 1 : 0; break;
            case 1: result = (v >= threshold) ? 1 : 0; break;
            case 2: result = (v <  threshold) ? 1 : 0; break;
            case 3: result = (v <= threshold) ? 1 : 0; break;
            case 4: result = (v == threshold) ? 1 : 0; break;
            case 5: result = (v != threshold) ? 1 : 0; break;
        }
        mask[i] = result;
    }
}

/* AND two masks together */
__global__ void mask_and_kernel(
    unsigned char *mask_a,
    const unsigned char *mask_b,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) mask_a[i] &= mask_b[i];
}

/* OR two masks together */
__global__ void mask_or_kernel(
    unsigned char *mask_a,
    const unsigned char *mask_b,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) mask_a[i] |= mask_b[i];
}

/* --- 2. EXCLUSIVE SCAN (Blelloch, per-block) ------------------------------- */
__global__ void exclusive_scan_block(
    const unsigned char *in,
    int *out,
    int N
){
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    temp[tid] = (i < N) ? (int)in[i] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1){
        int t = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += t;
        __syncthreads();
    }

    /* inclusive -> exclusive */
    int inclusive = temp[tid];
    if (i < N) out[i] = inclusive - (int)in[i];
}

/* Reduce each block to a single sum */
__global__ void block_reduce_sum(
    const unsigned char *in,
    int *block_sums,
    int N
){
    extern __shared__ int temp[];
    int tid        = threadIdx.x;
    int i          = blockIdx.x * blockDim.x + tid;
    temp[tid]      = (i < N) ? (int)in[i] : 0;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (tid < s) temp[tid] += temp[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = temp[0];
}

/* Add pre-computed block offsets to per-block exclusive scan -> global scan */
__global__ void add_block_offsets(int *scan, const int *offsets, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) scan[i] += offsets[blockIdx.x];
}

/* --- 3. SCATTER / COMPACT --------------------------------------------------- */
__global__ void scatter_compact(
    const unsigned char *mask,
    const int           *scan,
    const double        *col_in,
    const int           *grp_in,   /* group-by key (int32 dict-encoded, -1 = none) */
    double              *col_out,
    int                 *grp_out,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && mask[i]){
        int pos      = scan[i];
        col_out[pos] = col_in[i];
        grp_out[pos] = grp_in[i];
    }
}

/* --- 4. GROUP-BY AGGREGATIONS ----------------------------------------------- */

/* atomicAdd for double via CAS - works on all sm >= 2.0, no arch flag needed */
__device__ double atomicAddDouble(double *addr, double val){
    unsigned long long int *addr_ull = (unsigned long long int *)addr;
    unsigned long long int old_ull   = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old_ull;
        old_ull = atomicCAS(addr_ull, assumed,
                            __double_as_longlong(
                                __longlong_as_double(assumed) + val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}

/* SUM */
__global__ void groupby_sum(
    const int *grp, const double *val,
    double *agg, int M, int K
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M){
        int g = grp[i];
        if (g >= 0 && g < K) atomicAddDouble(&agg[g], val[i]);
    }
}

/* COUNT (mask already applied; every element counts as 1) */
__global__ void groupby_count(
    const int *grp,
    int *cnt, int M, int K
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M){
        int g = grp[i];
        if (g >= 0 && g < K) atomicAdd(&cnt[g], 1);
    }
}

/* MIN - uses double-precision atomicMin emulation via CAS */
__device__ double atomicMinDouble(double *addr, double val){
    unsigned long long int *addr_ull = (unsigned long long int *)addr;
    unsigned long long int old_ull   = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old_ull;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) break;
        old_ull = atomicCAS(addr_ull, assumed,
                            __double_as_longlong(val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}
__device__ double atomicMaxDouble(double *addr, double val){
    unsigned long long int *addr_ull = (unsigned long long int *)addr;
    unsigned long long int old_ull   = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old_ull;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break;
        old_ull = atomicCAS(addr_ull, assumed,
                            __double_as_longlong(val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}

__global__ void groupby_min(
    const int *grp, const double *val,
    double *agg, int M, int K
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M){
        int g = grp[i];
        if (g >= 0 && g < K) atomicMinDouble(&agg[g], val[i]);
    }
}

__global__ void groupby_max(
    const int *grp, const double *val,
    double *agg, int M, int K
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M){
        int g = grp[i];
        if (g >= 0 && g < K) atomicMaxDouble(&agg[g], val[i]);
    }
}

/* --- 5. FULL-COLUMN SUM / COUNT / MIN / MAX (no group-by) ------------------- */
__global__ void reduce_sum(const double *val, double *partial, int N){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? val[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void reduce_min(const double *val, double *partial, int N){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? val[i] : 1e300;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (tid < s) sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void reduce_max(const double *val, double *partial, int N){
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? val[i] : -1e300;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

} // extern "C"
"""

if GPU_AVAILABLE:
    try:
        # Context is still current on the main thread (autoinit pushed it above).
        # Compile kernels now, then pop once so worker threads can push/pop it.
        _mod = SourceModule(CUDA_SOURCE, options=[f"-arch={_cuda_arch}",
                            "-Wno-deprecated-gpu-targets"])
        _filter_kernel       = _mod.get_function("filter_kernel")
        _mask_and_kernel     = _mod.get_function("mask_and_kernel")
        _mask_or_kernel      = _mod.get_function("mask_or_kernel")
        _excl_scan           = _mod.get_function("exclusive_scan_block")
        _block_reduce        = _mod.get_function("block_reduce_sum")
        _add_offsets         = _mod.get_function("add_block_offsets")
        _scatter             = _mod.get_function("scatter_compact")
        _grp_sum             = _mod.get_function("groupby_sum")
        _grp_count           = _mod.get_function("groupby_count")
        _grp_min             = _mod.get_function("groupby_min")
        _grp_max             = _mod.get_function("groupby_max")
        _red_sum             = _mod.get_function("reduce_sum")
        _red_min             = _mod.get_function("reduce_min")
        _red_max             = _mod.get_function("reduce_max")
        # Kernels loaded — detach context so worker threads can own it via push/pop
        _cuda_context.pop()
    except Exception as e:
        GPU_AVAILABLE = False
        print(f"[WARN] CUDA compile failed: {e}. Falling back to CPU (NumPy).")


# ===============================================================================
#  COLUMNAR STORE  (SSD -> pinned RAM -> GPU)
# ===============================================================================

class ColumnarStore:
    """
    On-disk layout:
        <DATA_DIR>/<table_name>/
            _meta.json          - schema: {columns: [{name, dtype, dict_encoded}]}
            <col>.f64           - raw float64 binary (numeric / date epoch)
            <col>.i32           - raw int32 binary  (dict-encoded categoricals)
            <col>.dict.json     - id->string mapping for dict-encoded columns
    """

    def __init__(self, root: Path = DATA_DIR):
        self.root = root

    # -- Schema helpers --------------------------------------------------------
    def list_tables(self):
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "_meta.json").exists())

    def get_schema(self, table: str) -> dict:
        meta_path = self.root / table / "_meta.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    def list_columns(self, table: str):
        schema = self.get_schema(table)
        return schema.get("columns", [])

    # -- Load column into pinned host memory -----------------------------------
    def load_column_pinned(self, table: str, col_name: str):
        """Return (np_array, col_meta_dict).
        Numeric cols -> float64 contiguous array.
        Dict-encoded cols -> int32 contiguous array + dict returned in meta.
        Arrays are page-aligned contiguous so gpuarray.to_gpu uses a single DMA.
        """
        col_dir = self.root / table
        schema  = self.get_schema(table)
        col_meta = next((c for c in schema["columns"] if c["name"] == col_name), None)
        if col_meta is None:
            raise KeyError(f"Column '{col_name}' not found in '{table}'")

        if col_meta.get("dict_encoded"):
            path = col_dir / f"{col_name}.i32"
            raw  = np.frombuffer(path.read_bytes(), dtype=np.int32)
            arr  = np.ascontiguousarray(raw, dtype=np.int32)
            dict_path = col_dir / f"{col_name}.dict.json"
            col_meta["dict"] = json.loads(dict_path.read_text()) if dict_path.exists() else {}
        else:
            path = col_dir / f"{col_name}.f64"
            raw  = np.frombuffer(path.read_bytes(), dtype=np.float64)
            arr  = np.ascontiguousarray(raw, dtype=np.float64)

        return arr, col_meta

    # -- Create / populate demo table ------------------------------------------
    def create_demo_table(self, table: str = "sales", n_rows: int = 5_000_000):
        """Generate a demo table for testing when no real data exists."""
        tdir = self.root / table
        tdir.mkdir(exist_ok=True)

        rng = np.random.default_rng(42)

        # Columns
        amount   = rng.exponential(scale=200.0, size=n_rows).astype(np.float64)
        quantity = rng.integers(1, 100, size=n_rows).astype(np.float64)
        price    = rng.uniform(5.0, 500.0, size=n_rows).astype(np.float64)
        country_ids = rng.integers(0, 10, size=n_rows).astype(np.int32)
        country_dict = {str(i): c for i, c in enumerate(
            ["US","IN","DE","FR","GB","JP","CN","BR","AU","CA"])}

        # Write
        (tdir / "amount.f64").write_bytes(amount.tobytes())
        (tdir / "quantity.f64").write_bytes(quantity.tobytes())
        (tdir / "price.f64").write_bytes(price.tobytes())
        (tdir / "country.i32").write_bytes(country_ids.tobytes())
        (tdir / "country.dict.json").write_text(json.dumps(country_dict))

        meta = {
            "rows": n_rows,
            "columns": [
                {"name": "amount",   "dtype": "float64", "dict_encoded": False},
                {"name": "quantity", "dtype": "float64", "dict_encoded": False},
                {"name": "price",    "dtype": "float64", "dict_encoded": False},
                {"name": "country",  "dtype": "int32",   "dict_encoded": True},
            ]
        }
        (tdir / "_meta.json").write_text(json.dumps(meta, indent=2))
        return meta


store = ColumnarStore()


# ===============================================================================
#  GPU PIPELINE
# ===============================================================================
OP_MAP = {">": 0, ">=": 1, "<": 2, "<=": 3, "==": 4, "!=": 5}


def _grid(n, block=BLOCK_SIZE):
    return (n + block - 1) // block


def _global_scan_gpu(mask_d: "gpuarray", N: int):
    """Return (scan_d, total) - global exclusive prefix sum of mask_d."""
    g = _grid(N)
    sh = BLOCK_SIZE * np.dtype(np.int32).itemsize
    scan_d = gpuarray.empty(N, dtype=np.int32)

    _excl_scan(mask_d, scan_d, np.int32(N),
               block=(BLOCK_SIZE, 1, 1), grid=(g, 1, 1), shared=sh)

    blk_d = gpuarray.empty(g, dtype=np.int32)
    _block_reduce(mask_d, blk_d, np.int32(N),
                  block=(BLOCK_SIZE, 1, 1), grid=(g, 1, 1), shared=sh)

    blk_h = blk_d.get()
    off_h = np.zeros(g, dtype=np.int32)
    run = 0
    for i in range(g):
        off_h[i] = run
        run += blk_h[i]

    off_d = gpuarray.to_gpu(off_h)
    _add_offsets(scan_d, off_d, np.int32(N),
                 block=(BLOCK_SIZE, 1, 1), grid=(g, 1, 1))
    return scan_d, int(run)


def run_query_gpu(
    table: str,
    agg_col: str,
    agg_op: str,          # SUM | COUNT | MIN | MAX | AVG
    group_col: Optional[str],
    conditions: list,     # [{"col": str, "op": str, "val": float, "logic": "AND"|"OR"}]
    progress_cb=None,
) -> dict:
    """
    Full GPU pipeline. Must be called from a worker thread.
    Pushes the global CUDA context at entry and pops it on exit (even on error),
    so the calling thread temporarily owns the GPU context for the duration.
    """
    # -- Push CUDA context into this thread ------------------------------------
    _cuda_context.push()
    try:
        return _run_query_gpu_inner(table, agg_col, agg_op, group_col,
                                    conditions, progress_cb)
    finally:
        _cuda_context.pop()


def _run_query_gpu_inner(
    table, agg_col, agg_op, group_col, conditions, progress_cb
):
    schema = store.get_schema(table)
    N = schema["rows"]

    # -- Collect unique columns we need ---------------------------------------
    needed_cols = set([agg_col])
    if group_col:
        needed_cols.add(group_col)
    for c in conditions:
        needed_cols.add(c["col"])

    if progress_cb: progress_cb(5, "Loading columns from SSD...")

    cols = {}   # col_name -> (np_array, meta)
    for cn in needed_cols:
        cols[cn] = store.load_column_pinned(table, cn)

    if progress_cb: progress_cb(20, "Uploading columns to GPU...")

    # -- Upload to GPU (single DMA per column) --------------------------------
    gpu_cols = {}
    for cn, (arr, meta) in cols.items():
        gpu_cols[cn] = gpuarray.to_gpu(arr)   # arr is C-contiguous np array

    # -- Build filter mask ----------------------------------------------------
    if progress_cb: progress_cb(35, "Applying filters on GPU...")

    if conditions:
        cond0 = conditions[0]
        col_d = gpu_cols[cond0["col"]]
        col_f = col_d.astype(np.float64) if col_d.dtype == np.int32 else col_d
        mask_d = gpuarray.empty(N, dtype=np.uint8)
        _filter_kernel(col_f, mask_d, np.float64(float(cond0["val"])),
                       np.int32(OP_MAP.get(cond0["op"], 0)), np.int32(N),
                       block=(BLOCK_SIZE, 1, 1), grid=(_grid(N), 1, 1))

        for cond in conditions[1:]:
            col_d2 = gpu_cols[cond["col"]]
            col_f2 = col_d2.astype(np.float64) if col_d2.dtype == np.int32 else col_d2
            tmp_mask = gpuarray.empty(N, dtype=np.uint8)
            _filter_kernel(col_f2, tmp_mask, np.float64(float(cond["val"])),
                           np.int32(OP_MAP.get(cond["op"], 0)), np.int32(N),
                           block=(BLOCK_SIZE, 1, 1), grid=(_grid(N), 1, 1))
            if cond.get("logic", "AND") == "OR":
                _mask_or_kernel(mask_d, tmp_mask, np.int32(N),
                                block=(BLOCK_SIZE, 1, 1), grid=(_grid(N), 1, 1))
            else:
                _mask_and_kernel(mask_d, tmp_mask, np.int32(N),
                                 block=(BLOCK_SIZE, 1, 1), grid=(_grid(N), 1, 1))
    else:
        mask_d = gpuarray.empty(N, dtype=np.uint8)
        mask_d.fill(1)

    # -- Global prefix scan -> compact ----------------------------------------
    if progress_cb: progress_cb(50, "Computing prefix scan & compacting...")

    scan_d, M = _global_scan_gpu(mask_d, N)
    if M == 0:
        return {"rows": 0, "data": [], "agg_op": agg_op,
                "agg_col": agg_col, "group_col": group_col}

    agg_d = gpu_cols[agg_col]
    if agg_d.dtype != np.float64:
        agg_d = agg_d.astype(np.float64)

    grp_dict = None
    if group_col:
        grp_d_raw = gpu_cols[group_col]
        if grp_d_raw.dtype == np.int32:
            grp_d    = grp_d_raw
            grp_dict = cols[group_col][1].get("dict", {})
        else:
            # Numeric group-by: dict-encode on CPU, upload plain array
            grp_arr     = cols[group_col][0].astype(np.float64)
            unique_vals = np.unique(grp_arr)
            val_to_id   = {v: i for i, v in enumerate(unique_vals)}
            encoded     = np.vectorize(val_to_id.get)(grp_arr).astype(np.int32)
            grp_dict    = {str(i): str(v) for v, i in val_to_id.items()}
            grp_d       = gpuarray.to_gpu(np.ascontiguousarray(encoded))
    else:
        grp_d    = gpuarray.to_gpu(np.zeros(N, dtype=np.int32))
        grp_dict = {"0": "ALL"}

    # Scatter to compacted arrays
    agg_f_d = gpuarray.empty(M, dtype=np.float64)
    grp_f_d = gpuarray.empty(M, dtype=np.int32)
    _scatter(mask_d, scan_d, agg_d, grp_d, agg_f_d, grp_f_d, np.int32(N),
             block=(BLOCK_SIZE, 1, 1), grid=(_grid(N), 1, 1))

    # -- Aggregation ----------------------------------------------------------
    if progress_cb: progress_cb(70, f"Computing {agg_op} on GPU...")

    K = len(grp_dict) if grp_dict else 1

    if agg_op in ("SUM", "AVG"):
        agg_out = gpuarray.zeros(K, dtype=np.float64)
        _grp_sum(grp_f_d, agg_f_d, agg_out, np.int32(M), np.int32(K),
                 block=(BLOCK_SIZE, 1, 1), grid=(_grid(M), 1, 1))
        result_arr = agg_out.get()
        if agg_op == "AVG":
            cnt_out = gpuarray.zeros(K, dtype=np.int32)
            _grp_count(grp_f_d, cnt_out, np.int32(M), np.int32(K),
                       block=(BLOCK_SIZE, 1, 1), grid=(_grid(M), 1, 1))
            cnt_arr = cnt_out.get().astype(np.float64)
            result_arr = np.where(cnt_arr > 0, result_arr / cnt_arr, 0.0)

    elif agg_op == "COUNT":
        cnt_out = gpuarray.zeros(K, dtype=np.int32)
        _grp_count(grp_f_d, cnt_out, np.int32(M), np.int32(K),
                   block=(BLOCK_SIZE, 1, 1), grid=(_grid(M), 1, 1))
        result_arr = cnt_out.get().astype(np.float64)

    elif agg_op == "MIN":
        agg_out = gpuarray.empty(K, dtype=np.float64)
        agg_out.fill(np.float64(1e300))
        _grp_min(grp_f_d, agg_f_d, agg_out, np.int32(M), np.int32(K),
                 block=(BLOCK_SIZE, 1, 1), grid=(_grid(M), 1, 1))
        result_arr = agg_out.get()

    elif agg_op == "MAX":
        agg_out = gpuarray.empty(K, dtype=np.float64)
        agg_out.fill(np.float64(-1e300))
        _grp_max(grp_f_d, agg_f_d, agg_out, np.int32(M), np.int32(K),
                 block=(BLOCK_SIZE, 1, 1), grid=(_grid(M), 1, 1))
        result_arr = agg_out.get()

    else:
        result_arr = np.zeros(K, dtype=np.float64)

    # -- Assemble result rows -------------------------------------------------
    if progress_cb: progress_cb(90, "Collecting results...")

    rows = []
    for idx in range(K):
        label = grp_dict.get(str(idx), str(idx)) if grp_dict else "ALL"
        rows.append((label, result_arr[idx]))

    rows = [(k, v) for k, v in rows if abs(v) > 0 or agg_op == "COUNT"]
    rows.sort(key=lambda x: -x[1])

    return {
        "rows": M,
        "data": rows,
        "agg_op": agg_op,
        "agg_col": agg_col,
        "group_col": group_col,
    }


def run_query_cpu(
    table: str, agg_col: str, agg_op: str,
    group_col: Optional[str], conditions: list,
    progress_cb=None,
) -> dict:
    """NumPy fallback when GPU is not available."""
    schema = store.get_schema(table)
    N = schema["rows"]

    if progress_cb: progress_cb(10, "Loading columns (CPU mode)...")
    pinned_cols = {}
    needed = set([agg_col] + ([group_col] if group_col else []) +
                 [c["col"] for c in conditions])
    for cn in needed:
        pinned_cols[cn] = store.load_column_pinned(table, cn)

    if progress_cb: progress_cb(30, "Applying filters...")
    mask = np.ones(N, dtype=bool)
    for i, cond in enumerate(conditions):
        arr, _ = pinned_cols[cond["col"]]
        arr_f  = arr.astype(np.float64)
        th     = float(cond["val"])
        logic  = cond.get("logic", "AND")
        op     = cond["op"]
        ops_fn = {">": np.greater, ">=": np.greater_equal,
                  "<": np.less,    "<=": np.less_equal,
                  "==": np.equal,  "!=": np.not_equal}
        cond_mask = ops_fn[op](arr_f, th)
        if i == 0 or logic == "AND":
            mask &= cond_mask
        else:
            mask |= cond_mask

    M = mask.sum()
    if M == 0:
        return {"rows": 0, "data": [], "agg_op": agg_op,
                "agg_col": agg_col, "group_col": group_col}

    agg_arr, _ = pinned_cols[agg_col]
    agg_f = agg_arr.astype(np.float64)[mask]

    grp_dict = None
    if group_col:
        grp_arr, grp_meta = pinned_cols[group_col]
        grp_f = grp_arr[mask]
        if grp_meta.get("dict_encoded"):
            grp_dict = grp_meta.get("dict", {})
            keys = sorted(grp_dict.keys(), key=int)
        else:
            unique = np.unique(grp_f)
            grp_dict = {str(i): str(v) for i, v in enumerate(unique)}
            keys = list(grp_dict.keys())
    else:
        grp_f = np.zeros(int(M), dtype=np.int32)
        grp_dict = {"0": "ALL"}
        keys = ["0"]

    if progress_cb: progress_cb(70, f"Computing {agg_op}...")
    rows = []
    for k in keys:
        kid = int(k)
        label = grp_dict.get(k, k)
        sel   = grp_f == kid
        sub   = agg_f[sel]
        if len(sub) == 0 and agg_op != "COUNT":
            continue
        if agg_op == "SUM":   val = sub.sum()
        elif agg_op == "COUNT": val = float(len(sub))
        elif agg_op == "AVG":   val = sub.mean() if len(sub) else 0.0
        elif agg_op == "MIN":   val = sub.min() if len(sub) else 0.0
        elif agg_op == "MAX":   val = sub.max() if len(sub) else 0.0
        else: val = 0.0
        rows.append((label, val))

    rows.sort(key=lambda x: -x[1])
    return {"rows": int(M), "data": rows, "agg_op": agg_op,
            "agg_col": agg_col, "group_col": group_col}


def run_query(table, agg_col, agg_op, group_col, conditions, progress_cb=None):
    t0 = time.time()
    if GPU_AVAILABLE:
        result = run_query_gpu(table, agg_col, agg_op, group_col,
                               conditions, progress_cb)
    else:
        result = run_query_cpu(table, agg_col, agg_op, group_col,
                               conditions, progress_cb)
    result["time_ms"] = (time.time() - t0) * 1000
    return result


# ===============================================================================
#  GUI
# ===============================================================================

# -- Colour palette ------------------------------------------------------------
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_CARD    = "#21262d"
BG_INPUT   = "#0d1117"
ACCENT     = "#58a6ff"
ACCENT2    = "#3fb950"
WARN       = "#f78166"
TEXT_MAIN  = "#e6edf3"
TEXT_DIM   = "#8b949e"
TEXT_TINY  = "#6e7681"
BORDER     = "#30363d"


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


class GPUAnalyticsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPU Columnar Analytics Engine")
        self.geometry("1280x820")
        self.configure(bg=BG_DARK)
        self.minsize(1000, 680)

        # State
        self.current_table  = tk.StringVar()
        self.agg_col        = tk.StringVar()
        self.agg_op         = tk.StringVar(value="SUM")
        self.group_col      = tk.StringVar(value="-- none --")
        self.conditions     = []   # list of condition dicts
        self._query_thread  = None

        self._build_styles()
        self._build_layout()
        self._refresh_tables()

        # Create demo table if no data exists
        if not store.list_tables():
            self._log("No tables found. Creating demo 'sales' table (5M rows)...")
            store.create_demo_table("sales", 5_000_000)
            self._log("Demo table created.")
            self._refresh_tables()

    # -- TTK styles ------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=TEXT_MAIN,
                         font=("Inter", 11), borderwidth=0)

        style.configure("TNotebook",          background=BG_PANEL, borderwidth=0)
        style.configure("TNotebook.Tab",
                         background=BG_CARD, foreground=TEXT_DIM,
                         padding=[18, 8], font=("Inter", 10, "bold"),
                         focuscolor=BG_CARD)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", ACCENT)])

        style.configure("Card.TFrame",        background=BG_CARD,  relief="flat")
        style.configure("Panel.TFrame",       background=BG_PANEL, relief="flat")
        style.configure("Dark.TFrame",        background=BG_DARK,  relief="flat")

        style.configure("TCombobox",
                         fieldbackground=BG_INPUT, background=BG_CARD,
                         foreground=TEXT_MAIN, arrowcolor=ACCENT,
                         selectbackground=ACCENT, selectforeground=TEXT_MAIN,
                         borderwidth=1, relief="flat")
        style.map("TCombobox", fieldbackground=[("readonly", BG_INPUT)])

        style.configure("TProgressbar",
                         troughcolor=BG_CARD, background=ACCENT,
                         thickness=4, borderwidth=0)

        style.configure("Treeview",
                         background=BG_INPUT, foreground=TEXT_MAIN,
                         fieldbackground=BG_INPUT, rowheight=28,
                         borderwidth=0)
        style.configure("Treeview.Heading",
                         background=BG_CARD, foreground=ACCENT,
                         font=("Inter", 10, "bold"), relief="flat")
        style.map("Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG_DARK)])

    # -- Main layout -----------------------------------------------------------
    def _build_layout(self):
        # Top bar
        topbar = tk.Frame(self, bg=BG_PANEL, height=52)
        topbar.pack(fill=tk.X, side=tk.TOP)
        topbar.pack_propagate(False)

        gpu_badge = "GPU OK" if GPU_AVAILABLE else "CPU (no GPU)"
        gpu_color = ACCENT2 if GPU_AVAILABLE else WARN
        tk.Label(topbar, text="[GPU] GPU Analytics Engine",
                 font=("Inter", 15, "bold"),
                 bg=BG_PANEL, fg=TEXT_MAIN).pack(side=tk.LEFT, padx=20, pady=12)
        tk.Label(topbar, text=gpu_badge,
                 font=("Inter", 10, "bold"),
                 bg=gpu_color, fg=BG_DARK,
                 padx=10, pady=3).pack(side=tk.RIGHT, padx=20, pady=14)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill=tk.X)

        # Body: left sidebar + right area
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(body)
        self._build_main(body)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=BG_PANEL, width=270)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X)

        # -- Tables ------------------------------------------------------------
        self._section_label(sidebar, "TABLE")

        self.table_listbox = tk.Listbox(
            sidebar, bg=BG_INPUT, fg=TEXT_MAIN, selectbackground=ACCENT,
            selectforeground=BG_DARK, font=("Inter", 11),
            borderwidth=0, highlightthickness=0,
            activestyle="none", relief="flat")
        self.table_listbox.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.table_listbox.bind("<<ListboxSelect>>", self._on_table_select)

        btn_frame = tk.Frame(sidebar, bg=BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=12, pady=(0, 16))
        self._btn(btn_frame, "+ Import Folder",
                  self._import_table, ACCENT).pack(side=tk.LEFT)
        self._btn(btn_frame, "Demo",
                  self._create_demo, TEXT_DIM).pack(side=tk.LEFT, padx=(6,0))

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12)

        # -- Columns info ------------------------------------------------------
        self._section_label(sidebar, "SCHEMA")

        self.schema_tree = ttk.Treeview(sidebar,
                                         columns=("type",), show="headings",
                                         height=8)
        self.schema_tree.heading("type", text="Column  (type)")
        self.schema_tree.column("type", anchor="w")
        self.schema_tree.pack(fill=tk.X, padx=12, pady=(0, 12))

        # -- Status log -------------------------------------------------------
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12)
        self._section_label(sidebar, "LOG")

        self.log_box = tk.Text(
            sidebar, bg=BG_INPUT, fg=TEXT_DIM,
            font=("JetBrains Mono", 9), height=6,
            wrap=tk.WORD, state=tk.DISABLED,
            borderwidth=0, highlightthickness=0, relief="flat")
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def _build_main(self, parent):
        main = tk.Frame(parent, bg=BG_DARK)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Query builder card
        card = tk.Frame(main, bg=BG_PANEL, pady=0)
        card.pack(fill=tk.X, padx=16, pady=16)

        self._section_label(card, "QUERY BUILDER", padx=16)

        # Row 1: Aggregation
        r1 = tk.Frame(card, bg=BG_PANEL)
        r1.pack(fill=tk.X, padx=16, pady=(0, 10))

        self._label(r1, "Aggregate").pack(side=tk.LEFT)
        self.agg_op_cb = self._combobox(r1, self.agg_op,
                                         ["SUM", "COUNT", "MIN", "MAX", "AVG"],
                                         width=8)
        self.agg_op_cb.pack(side=tk.LEFT, padx=(8, 16))

        self._label(r1, "of column").pack(side=tk.LEFT)
        self.agg_col_cb = self._combobox(r1, self.agg_col, [], width=14)
        self.agg_col_cb.pack(side=tk.LEFT, padx=(8, 16))

        self._label(r1, "Group by").pack(side=tk.LEFT)
        self.group_col_cb = self._combobox(r1, self.group_col,
                                            ["-- none --"], width=14)
        self.group_col_cb.pack(side=tk.LEFT, padx=(8, 0))

        # Row 2: WHERE conditions
        r2 = tk.Frame(card, bg=BG_PANEL)
        r2.pack(fill=tk.X, padx=16, pady=(0, 12))

        self._label(r2, "WHERE").pack(side=tk.LEFT)
        self.cond_frame = tk.Frame(r2, bg=BG_PANEL)
        self.cond_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        add_btn = self._btn(r2, "+ Add condition", self._add_condition, ACCENT)
        add_btn.pack(side=tk.RIGHT)

        # Progress + Run
        r3 = tk.Frame(card, bg=BG_PANEL)
        r3.pack(fill=tk.X, padx=16, pady=(0, 14))

        self.run_btn = self._btn(r3, ">  Run Query", self._run_query, ACCENT2,
                                 font=("Inter", 12, "bold"), padx=20, pady=8)
        self.run_btn.pack(side=tk.LEFT)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(r3, variable=self.progress_var,
                                         maximum=100, style="TProgressbar",
                                         length=200)
        self.progress.pack(side=tk.LEFT, padx=16)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(r3, textvariable=self.status_var,
                 font=("Inter", 10), bg=BG_PANEL,
                 fg=TEXT_DIM).pack(side=tk.LEFT)

        # Results area (notebook)
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        # Tab 1: Table
        tab_table = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(tab_table, text="  Results Table  ")
        self._build_result_table(tab_table)

        # Tab 2: Summary
        tab_summary = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(tab_summary, text="  Summary  ")
        self._build_summary(tab_summary)

    def _build_result_table(self, parent):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        cols = ("group", "value")
        self.result_tree = ttk.Treeview(frame, columns=cols, show="headings")
        self.result_tree.heading("group", text="Group / Key")
        self.result_tree.heading("value", text="Aggregated Value")
        self.result_tree.column("group", anchor="w", width=300)
        self.result_tree.column("value", anchor="e", width=200)

        vsb = ttk.Scrollbar(frame, orient="vertical",
                             command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.pack(fill=tk.BOTH, expand=True)

    def _build_summary(self, parent):
        self.summary_text = tk.Text(
            parent, bg=BG_INPUT, fg=TEXT_MAIN,
            font=("JetBrains Mono", 11), wrap=tk.WORD,
            borderwidth=0, highlightthickness=0, relief="flat",
            state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.summary_text.tag_configure("title",  foreground=ACCENT,
                                         font=("JetBrains Mono", 12, "bold"))
        self.summary_text.tag_configure("kv",     foreground=ACCENT2)
        self.summary_text.tag_configure("dim",    foreground=TEXT_DIM)

    # -- Helpers ---------------------------------------------------------------
    def _section_label(self, parent, text, padx=12):
        tk.Label(parent, text=text,
                 font=("Inter", 9, "bold"),
                 bg=parent["bg"], fg=TEXT_TINY,
                 padx=padx).pack(anchor="w", pady=(14, 4))

    def _label(self, parent, text):
        return tk.Label(parent, text=text,
                        font=("Inter", 10), bg=BG_PANEL, fg=TEXT_DIM)

    def _combobox(self, parent, var, values, width=12):
        cb = ttk.Combobox(parent, textvariable=var,
                           values=values, state="readonly",
                           width=width, font=("Inter", 11))
        return cb

    def _btn(self, parent, text, cmd, color, font=("Inter", 10, "bold"),
             padx=12, pady=5):
        b = tk.Button(parent, text=text, command=cmd,
                      font=font, bg=color, fg=BG_DARK,
                      activebackground=BG_CARD, activeforeground=TEXT_MAIN,
                      relief="flat", bd=0, padx=padx, pady=pady,
                      cursor="hand2")
        return b

    def _log(self, msg):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    # -- Condition row ---------------------------------------------------------
    def _add_condition(self):
        idx = len(self.conditions)
        row = tk.Frame(self.cond_frame, bg=BG_PANEL)
        row.pack(anchor="w", pady=2)

        logic_var = tk.StringVar(value="AND" if idx > 0 else "WHERE")
        col_var   = tk.StringVar()
        op_var    = tk.StringVar(value=">")
        val_var   = tk.StringVar(value="0")

        schema = store.get_schema(self.current_table.get() or "")
        cols   = [c["name"] for c in schema.get("columns", [])] or [""]
        if col_var.get() == "" and cols:
            col_var.set(cols[0])

        if idx > 0:
            ttk.Combobox(row, textvariable=logic_var,
                          values=["AND", "OR"], state="readonly",
                          width=5, font=("Inter", 10)).pack(side=tk.LEFT, padx=(0,4))
        else:
            tk.Label(row, text="WHERE", font=("Inter", 10),
                     bg=BG_PANEL, fg=TEXT_DIM, width=6).pack(side=tk.LEFT)

        ttk.Combobox(row, textvariable=col_var, values=cols,
                      state="readonly", width=12,
                      font=("Inter", 10)).pack(side=tk.LEFT, padx=4)
        ttk.Combobox(row, textvariable=op_var,
                      values=[">", ">=", "<", "<=", "==", "!="],
                      state="readonly", width=4,
                      font=("Inter", 10)).pack(side=tk.LEFT, padx=4)
        tk.Entry(row, textvariable=val_var, width=10,
                 bg=BG_INPUT, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                 font=("Inter", 11), relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side=tk.LEFT, padx=4)

        def remove():
            row.destroy()
            self.conditions.remove(cdict)
        tk.Button(row, text="x", command=remove,
                  bg=BG_PANEL, fg=WARN, font=("Inter", 10),
                  relief="flat", bd=0, cursor="hand2").pack(side=tk.LEFT, padx=4)

        cdict = {"col": col_var, "op": op_var, "val": val_var,
                 "logic": logic_var}
        self.conditions.append(cdict)

    def _get_conditions(self):
        result = []
        for i, cdict in enumerate(self.conditions):
            result.append({
                "col":   cdict["col"].get(),
                "op":    cdict["op"].get(),
                "val":   cdict["val"].get(),
                "logic": cdict["logic"].get() if i > 0 else "AND",
            })
        return result

    # -- Table management ------------------------------------------------------
    def _refresh_tables(self):
        tables = store.list_tables()
        self.table_listbox.delete(0, tk.END)
        for t in tables:
            self.table_listbox.insert(tk.END, f"  {t}")

    def _on_table_select(self, event):
        sel = self.table_listbox.curselection()
        if not sel:
            return
        table = self.table_listbox.get(sel[0]).strip()
        self.current_table.set(table)
        schema = store.get_schema(table)
        cols   = schema.get("columns", [])

        # Update schema tree
        self.schema_tree.delete(*self.schema_tree.get_children())
        for c in cols:
            tag = "dict" if c.get("dict_encoded") else c.get("dtype", "")
            self.schema_tree.insert("", tk.END, values=(f"{c['name']}  [{tag}]",))

        # Update combos
        col_names = [c["name"] for c in cols]
        self.agg_col_cb["values"] = col_names
        if col_names:
            self.agg_col.set(col_names[0])
        self.group_col_cb["values"] = ["-- none --"] + col_names
        self.group_col.set("-- none --")

        rows = schema.get("rows", "?")
        self._log(f"Selected '{table}' ({rows:,} rows, {len(cols)} cols)")

    def _import_table(self):
        folder = filedialog.askdirectory(title="Select columnar table folder")
        if not folder:
            return
        p = Path(folder)
        if not (p / "_meta.json").exists():
            messagebox.showerror("Error",
                "Folder must contain '_meta.json' with schema.")
            return
        import shutil
        dest = DATA_DIR / p.name
        if dest.exists():
            if not messagebox.askyesno("Overwrite?",
                    f"Table '{p.name}' already exists. Overwrite?"):
                return
            shutil.rmtree(dest)
        shutil.copytree(p, dest)
        self._refresh_tables()
        self._log(f"Imported table '{p.name}'")

    def _create_demo(self):
        self._log("Creating demo table (5M rows)...")
        store.create_demo_table("sales", 5_000_000)
        self._refresh_tables()
        self._log("Demo 'sales' table ready.")

    # -- Query execution -------------------------------------------------------
    def _run_query(self):
        table = self.current_table.get()
        if not table:
            messagebox.showwarning("No table", "Please select a table first.")
            return

        agg_col = self.agg_col.get()
        agg_op  = self.agg_op.get()
        gc      = self.group_col.get()
        group_col = None if gc in ("-- none --", "") else gc
        conditions = self._get_conditions()

        if not agg_col:
            messagebox.showwarning("No column", "Select an aggregation column.")
            return

        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self.progress_var.set(0)
        self.status_var.set("Starting...")

        def _progress(pct, msg):
            self.progress_var.set(pct)
            self.status_var.set(msg)
            self.update_idletasks()

        def _worker():
            try:
                result = run_query(table, agg_col, agg_op, group_col,
                                   conditions, _progress)
                self.after(0, lambda: self._show_result(result))
            except Exception as e:
                self.after(0, lambda: self._show_error(str(e)))

        self._query_thread = threading.Thread(target=_worker, daemon=True)
        self._query_thread.start()

    def _show_result(self, result):
        self.run_btn.config(state=tk.NORMAL, text=">  Run Query")
        self.progress_var.set(100)

        # -- Results table tab -------------------------------------------------
        self.result_tree.delete(*self.result_tree.get_children())
        gc     = result.get("group_col") or "ALL"
        agg_op = result.get("agg_op", "")
        self.result_tree.heading("group", text=gc)
        self.result_tree.heading("value", text=f"{agg_op}({result.get('agg_col','')})")

        for i, (label, val) in enumerate(result["data"]):
            tag = "odd" if i % 2 else "even"
            self.result_tree.insert("", tk.END,
                                     values=(label, f"{val:,.4f}"), tags=(tag,))
        self.result_tree.tag_configure("odd",  background=BG_CARD)
        self.result_tree.tag_configure("even", background=BG_INPUT)

        # -- Summary tab -------------------------------------------------------
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)

        t_ms   = result.get("time_ms", 0)
        rows_in  = result.get("rows", 0)
        rows_out = len(result["data"])

        self.summary_text.insert(tk.END, "Query Summary\n", "title")
        self.summary_text.insert(tk.END, "-" * 50 + "\n", "dim")

        kvs = [
            ("Execution time",  f"{t_ms:.1f} ms"),
            ("Rows matched",    f"{rows_in:,}"),
            ("Groups returned", f"{rows_out:,}"),
            ("Aggregation",     f"{agg_op}({result.get('agg_col','')})"),
            ("Group by",        result.get("group_col") or "none"),
            ("Backend",         "CUDA GPU" if GPU_AVAILABLE else "NumPy CPU"),
        ]
        for k, v in kvs:
            self.summary_text.insert(tk.END, f"  {k:<20}", "dim")
            self.summary_text.insert(tk.END, f"{v}\n", "kv")

        if result["data"]:
            self.summary_text.insert(tk.END, "\nTop results:\n", "title")
            for label, val in result["data"][:10]:
                self.summary_text.insert(tk.END,
                    f"  {str(label):<24}  {val:>16,.4f}\n")

        self.summary_text.config(state=tk.DISABLED)
        self.nb.select(0)

        self.status_var.set(
            f"Done -- {rows_in:,} rows matched, {t_ms:.1f} ms")
        self._log(f"Query OK: {agg_op}({result.get('agg_col','')}) "
                  f"in {t_ms:.1f} ms -> {rows_out} groups")

    def _show_error(self, msg):
        self.run_btn.config(state=tk.NORMAL, text=">  Run Query")
        self.progress_var.set(0)
        self.status_var.set(f"Error: {msg[:60]}")
        self._log(f"ERROR: {msg}")
        messagebox.showerror("Query Error", msg)


# ===============================================================================
if __name__ == "__main__":
    app = GPUAnalyticsApp()
    app.mainloop()