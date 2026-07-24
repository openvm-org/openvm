#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    let manifest_dir = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"),
    );
    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let abi_path = manifest_dir.join("tracegen_abi.def");
    println!("cargo:rerun-if-changed={}", abi_path.display());
    let abi = std::fs::read_to_string(&abi_path).expect("read tracegen_abi.def");
    let mut rust = String::from("// @generated from tracegen_abi.def; do not edit.\n");
    let mut cuda = String::from(
        "// @generated from tracegen_abi.def; do not edit.\n#pragma once\n#include <cstdint>\n\n",
    );
    for (line_no, line) in abi.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let name = fields.next().expect("ABI constant name");
        let value = fields.next().expect("ABI constant value");
        assert!(
            fields.next().is_none(),
            "invalid tracegen ABI line {}",
            line_no + 1
        );
        value
            .parse::<u32>()
            .unwrap_or_else(|_| panic!("invalid tracegen ABI value on line {}", line_no + 1));
        rust.push_str(&format!("pub const {name}: u32 = {value};\n"));
        cuda.push_str(&format!("static constexpr uint32_t {name} = {value};\n"));
    }
    std::fs::write(out_dir.join("tracegen_abi.rs"), rust)
        .expect("write generated Rust tracegen ABI");
    std::fs::write(out_dir.join("tracegen_abi.cuh"), cuda)
        .expect("write generated CUDA tracegen ABI");

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder = CudaBuilder::new()
            .include(&out_dir)
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("../primitives/cuda/include")
            .include("../../vm/cuda/include")
            .include("../../../extensions/riscv/circuit/cuda/include")
            .include("../../../extensions/riscv-adapters/cuda/include")
            .watch("cuda")
            .watch("../primitives/cuda")
            .watch("../../../extensions/riscv-adapters/cuda")
            .library_name("tracegen_gpu_mod_builder")
            .file("cuda/src/field_expr.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
