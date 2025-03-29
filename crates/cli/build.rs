fn main() {
    vergen::EmitBuilder::builder()
        .build_timestamp()
        .git_sha(true)
        .emit()
        .unwrap();

    // Add the workspace version to the build
    println!(
        "cargo:rustc-env=OPENVM_WORKSPACE_VERSION={}",
        env!("CARGO_PKG_VERSION")
    );
}
