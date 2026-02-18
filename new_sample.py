import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

import psycopg2
import psycopg2.extras

import numpy as np
import pycuda.autoinit  # noqa: F401  # initializes CUDA context
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


# ==========================
# CONFIG
# ==========================

DB_DSN = "postgresql://postgres:postgres@localhost:5432/final_year_project"  
TABLE_NAME = "sales"
COUNTRY_DICT_TABLE = "country_dim"  # optional, if you have id->name mapping
GPU_MEM_BYTES = 4 * 1024 * 1024 * 1024  # 4GB RTX 2050
SAFETY_FACTOR = 0.5  # use only 50% of GPU mem for data
MAX_COUNTRIES = 1024  # assume dictionary-encoded country ids in [0, MAX_COUNTRIES)


# ==========================
# CUDA KERNELS
# ==========================

cuda_source = r"""
extern "C" {

__global__ void filter_kernel(
    const float *amount,
    unsigned char *mask,
    float threshold,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        float v = amount[i];
        mask[i] = (v > threshold) ? 1 : 0;
    }
}

// Blelloch scan per block (exclusive), then add block offsets.
// For simplicity, this version assumes N is multiple of blockDim.x.
// For production, handle arbitrary N more carefully.

__global__ void exclusive_scan_block(
    const unsigned char *in,
    int *out,
    int N
){
    extern __shared__ int temp[];  // 2*blockDim.x if doing 2 elements per thread; here 1 per thread

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // load into shared
    int val = 0;
    if (i < N){
        val = (int)in[i];
    }
    temp[tid] = val;
    __syncthreads();

    // upsweep
    for (int offset = 1; offset < blockDim.x; offset <<= 1){
        int t = 0;
        if (tid >= offset){
            t = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += t;
        __syncthreads();
    }

    // temp[tid] now holds inclusive scan.
    // convert to exclusive: shift right by 1, first element 0
    int inclusive = temp[tid];
    int exclusive = inclusive - val;
    if (i < N){
        out[i] = exclusive;
    }
}

// Compute block sums to add later (per-block correction)
__global__ void compute_block_sums(
    const unsigned char *in,
    int *block_sums,
    int N
){
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int i = blockStart + tid;

    int val = 0;
    if (i < N){
        val = (int)in[i];
    }
    temp[tid] = val;
    __syncthreads();

    // reduction to get sum of this block
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if (tid < offset){
            temp[tid] += temp[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0){
        block_sums[blockIdx.x] = temp[0];
    }
}

// Add scanned block sums to each element to get global exclusive scan
__global__ void add_block_offsets(
    int *scan,
    const int *block_offsets,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        int block_id = blockIdx.x;
        int offset = block_offsets[block_id];
        scan[i] += offset;
    }
}

// Scatter filtered rows into compacted arrays.
__global__ void scatter_compact(
    const unsigned char *mask,
    const int *scan,
    const int *country_in,
    const float *amount_in,
    int *country_out,
    float *amount_out,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        if (mask[i]){
            int pos = scan[i];
            country_out[pos] = country_in[i];
            amount_out[pos] = amount_in[i];
        }
    }
}

// Group-by country (0..K-1) with SUM(amount).
__global__ void groupby_sum_country(
    const int *country,
    const float *amount,
    double *agg_sum,
    int M
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M){
        int c = country[i];
        float v = amount[i];
        if (c >= 0){  // basic guard
            atomicAdd(&agg_sum[c], (double)v);
        }
    }
}

} // extern "C"
"""

mod = SourceModule(cuda_source)
filter_kernel = mod.get_function("filter_kernel")
exclusive_scan_block = mod.get_function("exclusive_scan_block")
compute_block_sums = mod.get_function("compute_block_sums")
add_block_offsets = mod.get_function("add_block_offsets")
scatter_compact = mod.get_function("scatter_compact")
groupby_sum_country = mod.get_function("groupby_sum_country")


# ==========================
# DB + CHUNKING HELPERS
# ==========================

def get_connection():
    # psycopg2 supports URL DSN directly.[web:101][web:107]
    conn = psycopg2.connect(DB_DSN)
    return conn


def estimate_row_count(conn, amount_threshold):
    # Run EXPLAIN to get estimated rows for this specific WHERE.[web:104]
    sql = """
    EXPLAIN SELECT country, amount
    FROM {table}
    WHERE amount > %s
    """.format(table=TABLE_NAME)
    with conn.cursor() as cur:
        cur.execute(sql, (amount_threshold,))
        plan_lines = [row[0] for row in cur.fetchall()]
    # Very naive: parse first line for "rows=X"
    # Example: "Seq Scan on sales  (cost=0.00..123.45 rows=100000 width=12)"
    est_rows = 0
    if plan_lines:
        line = plan_lines[0]
        import re
        m = re.search(r"rows=([0-9]+)", line)
        if m:
            est_rows = int(m.group(1))
    return est_rows


def compute_chunk_rows(est_rows):
    # Simple heuristic: based on GPU memory and row width.
    # country: int32 (4 bytes), amount: float32 (4 bytes) => ~8 bytes/row
    bytes_per_row = 8
    est_rows = 20000000
    usable_mem = int(GPU_MEM_BYTES * SAFETY_FACTOR)
    max_rows_by_mem = usable_mem // bytes_per_row
    if max_rows_by_mem <= 0:
        max_rows_by_mem = 1_000_000
    # But also cap by estimated rows to avoid too many tiny chunks
    chunk_rows = min(max_rows_by_mem, max(est_rows // 4, 1_000_000))
    # print("est_rows", est_rows)
    # print("max_rows_by_mem", max_rows_by_mem)
    # print("chunk_rows", int(chunk_rows))
    return int(min(max_rows_by_mem, est_rows))


def fetch_chunk(conn, offset, limit, amount_threshold):
    # Fetch chunk as dictionary-encoded country + amount
    # Assumes 'country' is already an int id in DB; if not, you will need
    # a separate dimension table + mapping.
    sql = f"""
    SELECT country, amount
    FROM {TABLE_NAME}
    WHERE amount > %s
    ORDER BY country, amount
    OFFSET %s
    LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (amount_threshold, offset, limit))
        rows = cur.fetchall()
    if not rows:
        return None, None
    # Convert to numpy arrays
    data = np.array(rows, dtype=[("country", np.int32), ("amount", np.float32)])
    country = data["country"].astype(np.int32)
    amount = data["amount"].astype(np.float32)
    return country, amount


# ==========================
# GPU PIPELINE FOR ONE CHUNK
# ==========================

def process_chunk_gpu(country_h, amount_h, amount_threshold):
    N = country_h.shape[0]

    # Upload to GPU
    country_d = gpuarray.to_gpu(country_h)
    amount_d = gpuarray.to_gpu(amount_h)

    # 1) filter -> mask
    mask_d = gpuarray.empty(N, dtype=np.uint8)

    block = 256
    grid = (N + block - 1) // block

    filter_kernel(
        amount_d,
        mask_d,
        np.float32(amount_threshold),
        np.int32(N),
        block=(block, 1, 1),
        grid=(grid, 1, 1),
    )

    # 2) exclusive scan of mask to get positions
    scan_d = gpuarray.empty(N, dtype=np.int32)

    # per-block scan (exclusive) in shared mem
    shared_bytes = block * np.dtype(np.int32).itemsize
    exclusive_scan_block(
        mask_d,
        scan_d,
        np.int32(N),
        block=(block, 1, 1),
        grid=(grid, 1, 1),
        shared=shared_bytes,
    )

    # compute block sums (sum of mask within each block)
    block_sums_d = gpuarray.empty(grid, dtype=np.int32)
    compute_block_sums(
        mask_d,
        block_sums_d,
        np.int32(N),
        block=(block, 1, 1),
        grid=(grid, 1, 1),
        shared=shared_bytes,
    )

    # scan block_sums on CPU for simplicity (few blocks)
    block_sums_h = block_sums_d.get()
    block_offsets_h = np.zeros_like(block_sums_h)
    running = 0
    for i in range(grid):
        block_offsets_h[i] = running
        running += block_sums_h[i]

    total_kept = int(running)

    if total_kept == 0:
        # nothing passes filter
        return np.zeros(MAX_COUNTRIES, dtype=np.float64)

    block_offsets_d = gpuarray.to_gpu(block_offsets_h)

    add_block_offsets(
        scan_d,
        block_offsets_d,
        np.int32(N),
        block=(block, 1, 1),
        grid=(grid, 1, 1),
    )

    # 3) scatter to compact arrays
    country_f_d = gpuarray.empty(total_kept, dtype=np.int32)
    amount_f_d = gpuarray.empty(total_kept, dtype=np.float32)

    scatter_compact(
        mask_d,
        scan_d,
        country_d,
        amount_d,
        country_f_d,
        amount_f_d,
        np.int32(N),
        block=(block, 1, 1),
        grid=(grid, 1, 1),
    )

    # 4) group-by SUM by country id into fixed-size agg array
    agg_sum_d = gpuarray.zeros(MAX_COUNTRIES, dtype=np.float64)

    grid2 = (total_kept + block - 1) // block
    groupby_sum_country(
        country_f_d,
        amount_f_d,
        agg_sum_d,
        np.int32(total_kept),
        block=(block, 1, 1),
        grid=(grid2, 1, 1),
    )

    # return partial sums to host
    return agg_sum_d.get()


def run_full_query(amount_threshold: float):
    conn = get_connection()
    import time
    start_time = time.time_ns()
    try:
        est_rows = estimate_row_count(conn, amount_threshold)
        if est_rows <= 0:
            est_rows = 1_000_000  # fallback heuristic

        chunk_rows = compute_chunk_rows(est_rows)

        # Iterate chunks
        offset = 0
        global_sum = np.zeros(MAX_COUNTRIES, dtype=np.float64)

        while True:
            country_h, amount_h = fetch_chunk(conn, offset, chunk_rows, amount_threshold)
            if country_h is None:
                break

            partial_sum = process_chunk_gpu(country_h, amount_h, amount_threshold)
            global_sum += partial_sum

            offset += chunk_rows

        # Optionally, map country ids -> names from a dim table
        # For now, return non-zero entries as (country_id, sum)
        result = [
            (cid, val) for cid, val in enumerate(global_sum) if val != 0.0
        ]
        print(time.time_ns() - start_time)

        return result

    finally:
        conn.close()


# ==========================
# TKINTER GUI (YOUR CODE + HOOK)
# ==========================

# Create main window
root = tk.Tk()
root.title("Query Interface")
root.geometry("800x600")
root.configure(bg="#2b2b2b")

# Custom fonts
title_font = tkFont.Font(family="Segoe UI", size=16, weight="bold")
label_font = tkFont.Font(family="Segoe UI", size=12)
button_font = tkFont.Font(family="Segoe UI", size=11)

# Main container with sidebar and content
main_frame = tk.Frame(root, bg="#2b2b2b")
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left sidebar
sidebar = tk.Frame(main_frame, width=200, bg="#1e1e1e", relief=tk.RAISED, bd=1)
sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
sidebar.pack_propagate(False)

tk.Label(sidebar, text="Query Interface", font=title_font, bg="#1e1e1e", fg="#ffffff").pack(pady=20)
tk.Label(sidebar, text="• Modern Design", font=label_font, bg="#1e1e1e", fg="#cccccc").pack(anchor="w", padx=20, pady=5)
tk.Label(sidebar, text="• Clean Layout", font=label_font, bg="#1e1e1e", fg="#cccccc").pack(anchor="w", padx=20, pady=2)
tk.Label(sidebar, text="• Responsive", font=label_font, bg="#1e1e1e", fg="#cccccc").pack(anchor="w", padx=20, pady=2)

# Right content area
content_frame = tk.Frame(main_frame, bg="#2b2b2b")
content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Input area
input_frame = tk.Frame(content_frame, bg="#363636", relief=tk.RAISED, bd=1)
input_frame.pack(fill=tk.X, pady=(0, 10))

title_label = tk.Label(
    input_frame,
    text="Enter your query:",
    font=title_font,
    bg="#363636",
    fg="#ffffff",
    anchor="w",
)
title_label.pack(anchor="w", padx=20, pady=(20, 10))

text_entry = tk.Text(
    input_frame,
    height=4,
    font=("Segoe UI", 11),
    bg="#1e1e1e",
    fg="#ffffff",
    insertbackground="#ffffff",
    relief=tk.FLAT,
    bd=1,
    wrap=tk.WORD,
)
text_entry.pack(fill=tk.X, padx=20, pady=(0, 20))

# Results notebook
notebook = ttk.Notebook(content_frame)
notebook.pack(fill=tk.BOTH, expand=True)

# Results page
result_frame = tk.Frame(notebook, bg="#2b2b2b")
notebook.add(result_frame, text="Results")

# Result display area
result_text = tk.Text(
    result_frame,
    font=("Segoe UI", 11),
    bg="#1e1e1e",
    fg="#ffffff",
    insertbackground="#ffffff",
    relief=tk.FLAT,
    bd=1,
    wrap=tk.WORD,
    state=tk.DISABLED,
)
result_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)


def write_to_result(data, clear_first=False):
    result_text.config(state=tk.NORMAL)
    if clear_first:
        result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, data)
    result_text.config(state=tk.DISABLED)
    result_text.see(tk.END)


def parse_amount_threshold(query: str) -> float:
    """
    Very naive parser that extracts the number after 'amount >'.
    Assumes user only changes that literal in:
    SELECT country, SUM(amount) FROM sales WHERE amount > X GROUP BY country
    """
    import re
    m = re.search(r"amount\s*>\s*([0-9.]+)", query, re.IGNORECASE)
    if not m:
        raise ValueError("Could not parse amount threshold from query.")
    return float(m.group(1))


def show_result():
    query = text_entry.get("1.0", tk.END).strip()

    write_to_result(f"User Query:\n{query}\n\n", clear_first=True)
    write_to_result("Processing query on GPU...\n\n")

    try:
        amount_threshold = parse_amount_threshold(query)

        results = run_full_query(amount_threshold)

        write_to_result("Results (country_id, sum_amount):\n")
        for cid, s in results:
            line = f"- country_id={cid}, sum_amount={s:.2f}\n"
            write_to_result(line)
    except Exception as e:
        write_to_result(f"Error: {e}\n")


submit_btn = tk.Button(
    input_frame,
    text="Submit Query",
    font=button_font,
    bg="#007acc",
    fg="white",
    activebackground="#005a9e",
    relief=tk.FLAT,
    bd=0,
    height=2,
    command=show_result,
)
submit_btn.pack(anchor="w", padx=20, pady=(0, 20))

# Initial message
write_to_result("Ready to receive queries...\n")

if __name__ == "__main__":
    root.mainloop()

# SELECT country, SUM(amount) FROM sales WHERE amount > 100 GROUP BY country
