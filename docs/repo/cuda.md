# CUDA Implementation

This document describes the implementation and organization of CUDA GPU acceleration code throughout the OpenVM framework.

See [Development with CUDA](../contributor-setup.md#development-with-cuda) for more information on machine and IDE setup.

## Overview

The OpenVM framework includes optional GPU acceleration via CUDA for performance-critical components. GPU implementations are available as an optional feature `cuda` and can significantly speed up proof and trace generation.

## Project Structure

### Directory Organization

GPU-enabled crates in OpenVM generally follow a consistent layout:

```
crate-root/
├── src/                 # Rust source code
│   ├── module/          # Module with CUDA support
│   │   └── cuda(.rs)    # file or folder with Rust impl that uses CUDA
│   └── cuda_abi.rs      # FFI bindings to CUDA functions
├── cuda/                # CUDA implementation
│   ├── include/   
│   │   └── crate_name/  # Header files (.cuh, .h)
│   └── src/             # CUDA source files (.cu)
└── build.rs             # Build script using openvm-cuda-builder
└── Cargo.toml           # [build-dependencies] openvm-cuda-builder; [features] cuda

```

### Key Components

1. **CUDA Source Files** (`cuda/src/*.cu`)
   - Contain CUDA kernels
   - Include `extern "C"` launcher functions for kernel invocation

2. **Header Files** (`cuda/include/crate_name/*.cuh, *.h`)
   - CUDA declarations and common implementations
   - Organized by crate name for a proper namespaced include layout

3. **Rust FFI Binding File** (`src/cuda_abi.rs`)
   - Maps Rust functions to CUDA `extern "C"` launchers
   - Provides a safe Rust interface for CUDA functionality

4. **CUDA Support Module** (`src/cuda.rs` or `src/cuda/`)
   - Rust code supporting CUDA implementation
   - Only included when `cuda` feature is enabled via conditional compilation

5. **Build Configuration** (`build.rs`)
   - Uses [`openvm-cuda-builder`](https://github.com/openvm-org/stark-backend/tree/main/crates/cuda-builder) for CUDA compilation
   - Must include `openvm-cuda-builder` in `[build-dependencies]`

## Builder Pattern

Extensions with both CPU and GPU implementations follow a consistent builder pattern:

- `...CpuBuilder` - CPU implementation builder
- `...GpuBuilder` - GPU implementation builder  
- `...Builder` - Public alias that resolves to either CPU or GPU builder based on whether or not the `cuda` feature flag is set

Example:
```rust
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::*;
        pub use cuda::{
            Keccak256GpuProverExt as Keccak256ProverExt,
            Keccak256Rv32GpuBuilder as Keccak256Rv32Builder,
        };
    } else {
        pub use self::{
            Keccak256CpuProverExt as Keccak256ProverExt,
            Keccak256Rv32CpuBuilder as Keccak256Rv32Builder,
        };
    }
}
```
