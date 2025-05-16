use std::path::PathBuf;

use cargo_openvm::{
    commands::{build, BuildArgs, BuildCargoArgs},
    default::{DEFAULT_APP_EXE_PATH, DEFAULT_COMMITTED_APP_EXE_PATH},
};
use eyre::Result;
use openvm_build::RUSTC_TARGET;
use openvm_circuit::arch::OPENVM_DEFAULT_INIT_FILE_NAME;

fn default_build_args(example: &str) -> BuildArgs {
    BuildArgs {
        no_transpile: true,
        config: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("programs")
            .join(example)
            .join("openvm.toml"),
        exe_output: PathBuf::from(DEFAULT_APP_EXE_PATH),
        committed_exe_output: PathBuf::from(DEFAULT_COMMITTED_APP_EXE_PATH),
        init_file_name: OPENVM_DEFAULT_INIT_FILE_NAME.to_string(),
    }
}

fn default_cargo_args(example: &str) -> BuildCargoArgs {
    BuildCargoArgs {
        package: vec![],
        workspace: false,
        exclude: vec![],
        lib: false,
        bin: vec![],
        bins: false,
        example: vec![],
        examples: false,
        all_targets: false,
        all_features: false,
        no_default_features: false,
        features: vec![],
        profile: "release".to_string(),
        target_dir: None,
        verbose: false,
        quiet: false,
        color: "always".to_string(),
        manifest_path: Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("programs")
                .join(example)
                .join("Cargo.toml"),
        ),
        ignore_rust_version: false,
        locked: false,
        offline: false,
        frozen: false,
    }
}

#[test]
fn test_build_with_profile() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let target_dir = temp_dir.path();

    let build_args = default_build_args("fibonacci");
    let mut cargo_args = default_cargo_args("fibonacci");
    cargo_args.target_dir = Some(target_dir.to_path_buf());
    cargo_args.profile = "dev".to_string();

    build(&build_args, &cargo_args)?;
    assert!(
        target_dir.join(RUSTC_TARGET).join("debug").exists(),
        "did not build with dev profile"
    );
    temp_dir.close()?;
    Ok(())
}

#[test]
fn test_multi_target_build() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let target_dir = temp_dir.path();

    let build_args = default_build_args("multi");
    let mut cargo_args = default_cargo_args("multi");
    cargo_args.target_dir = Some(target_dir.to_path_buf());

    // Build lib
    cargo_args.lib = true;
    let build_result = build(&build_args, &cargo_args)?;
    assert!(build_result.is_empty());

    // Build bins
    cargo_args.lib = false;
    let build_result = build(&build_args, &cargo_args)?;
    let binary_names: Vec<String> = build_result
        .iter()
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(binary_names.len() == 2);
    assert!(binary_names.contains(&"calculator".to_string()));
    assert!(binary_names.contains(&"string-utils".to_string()));

    // Build examples
    cargo_args.examples = true;
    let build_result = build(&build_args, &cargo_args)?;
    let example_names: Vec<String> = build_result
        .iter()
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(example_names.len() == 2);
    assert!(example_names.contains(&"fibonacci".to_string()));
    assert!(example_names.contains(&"palindrome".to_string()));

    // Build examples and a single bin
    cargo_args.bin = vec!["calculator".to_string()];
    let build_result = build(&build_args, &cargo_args)?;
    let exe_names: Vec<String> = build_result
        .iter()
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(exe_names.len() == 3);
    assert!(exe_names.contains(&"calculator".to_string()));
    assert!(exe_names.contains(&"fibonacci".to_string()));
    assert!(exe_names.contains(&"palindrome".to_string()));

    // Build all targets
    cargo_args.bin = vec![];
    cargo_args.examples = false;
    cargo_args.all_targets = true;
    let build_result = build(&build_args, &cargo_args)?;
    let all_names: Vec<String> = build_result
        .iter()
        .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(all_names.len() == 4);
    assert!(all_names.contains(&"calculator".to_string()));
    assert!(all_names.contains(&"string-utils".to_string()));
    assert!(all_names.contains(&"fibonacci".to_string()));
    assert!(all_names.contains(&"palindrome".to_string()));

    Ok(())
}
