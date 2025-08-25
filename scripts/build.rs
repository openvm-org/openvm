use std::env::var;

fn main() {
    if let Ok(v) = var("DEP_CUDA_COMMON_INCLUDE") {
        println!("cargo:rustc-env=DEP_CUDA_COMMON_INCLUDE={v}");
    }
}
