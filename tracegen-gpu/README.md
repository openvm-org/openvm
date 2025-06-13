# GPU Trace Generation

## CUDA

All CUDA code is kept in the `cuda/` directory. Each kernel has an associated `extern "C" ...` launcher, which is usable in Rust.

### Accessing Trace Values

When filling out a trace, sometimes it is convenient to fill consecutive columns within a single row together. However, because the trace is a column-major matrix on the GPU, consecutive row values do not correspond to consecutive places in memory as they do in OpenVM.

Utility struct `RowSlice` is defined in `cuda/include/trace_access.h` in order to resolve this problem. It contains a pointer to the first element of the row section, and the distance between consecutive row values (i.e. `stride`). For example, to access the first 4 columns given a particular row we do the following:

```
__global__ void some_cukernel(Fp *trace, uint32_t trace_height, ...) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    ...

    RowSlice row(trace + idx, trace_height);
    row[0] = ...;
    row[1] = ...;
    row[2] = ...;
    row[3] = ...;
}
```

To make it so we can write trace values without manually indexing where each column is, `cuda/include/trace_access.h` also contains several macros that allow a `RowSlice` to be treated as a `*Cols` struct with named fields.

```
template <typename T> struct SomeCols {
    T col_a;
    T col_b;
    T col_array[4];
}

__global__ void some_cukernel(Fp *trace, uint32_t trace_height, ...) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, trace_height);

    ...

    COL_WRITE_VALUE(row, SomeCols, col_a, some_value_a);
    COL_WRITE_VALUE(row, SomeCols, col_b, some_value_b);
    COL_WRITE_ARRAY(row, SomeCols, col_array, some_array);
}
```

### SubAirs

OpenVM's `SubAir` trait enforces constraints on a row slice instead of the entire row - they enable AIRs that want to constrain common properties about some of their values (ex. whether or not some number `x` is less than some `y`) to do so without code duplication. To this effect, each AIR's trace matrix must contain a subset of columns that correspond to the `SubAir`'s functionality.

To help with trace generation most `SubAir`'s also implement `TraceSubRowGenerator`, which contains function `generate_subrow`. A chip can pass a mutable row slice to `generate_subrow` alongside some context, which will then populate those values according to the `SubAir`'s constraints.

In Axiom GPU's trace generation, `SubAir`s are provided in header `.cuh` files in `cuda/include/` represented as namespaces with a `generate_subrow` function. To use some `SubAir::generate_subrow`, read its documentation and fill in its arguments accordingly.

## Rust Crate

TODO
