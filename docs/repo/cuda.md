## CUDA Implementation

This document describes the CUDA GPU acceleration structure across the OpenVM framework.
See [Development with CUDA](../contributor-setup.md#development-with-cuda) for more information on machine and IDE setup.

### Overview

The OpenVM framework includes optional GPU acceleration via CUDA for performance-critical components. GPU implementations are available as an optional feature and can significantly speed up proof and trace generation.

### Project Structure

#### Directory Organization

Each crate with GPU implementation follows a consistent structure:

```
crate-root/
├── src/                 # Rust source code
│   ├── cuda.rs          # or cuda/ folder - CUDA support module
│   └── cuda_abi.rs      # FFI bindings to CUDA functions
├── cuda/                # CUDA implementation
│   ├── include/   
│   │   └── crate_name/  # Header files (.cuh, .h)
│   └── src/             # CUDA source files (.cu)
└── build.rs             # Build script using openvm-cuda-builder
```

#### Key Components

1. **CUDA Source Files** (`cuda/src/*.cu`)
   - Contains CUDA kernels
   - Includes `extern "C"` launcher functions for kernel invocation

2. **Header Files** (`cuda/include/crate_name/*.cuh, *.h`)
   - CUDA header files and declarations
   - Organized by crate name for proper namespacing

3. **Rust FFI Bindings** (`src/cuda_abi.rs`)
   - Maps between Rust functions and CUDA `extern "C"` launchers
   - Provides safe Rust interface to CUDA functionality

4. **CUDA Support Module** (`src/cuda.rs` or `src/cuda/`)
   - Rust code supporting CUDA implementation
   - Only included when `cuda` feature is enabled via conditional compilation

5. **Build Configuration** (`build.rs`)
   - Uses [`openvm-cuda-builder`](https://github.com/openvm-org/stark-backend/tree/main/crates/cuda-builder) for CUDA compilation
   - Must include `openvm-cuda-builder` in `[build-dependencies]`

### Builder Pattern

Extensions with both CPU and GPU implementations follow a consistent builder pattern:

- `...CpuBuilder` - CPU implementation builder
- `...GpuBuilder` - GPU implementation builder  
- `...Builder` - Public alias that resolves to either CPU or GPU builder based on the `cuda` feature flag

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
