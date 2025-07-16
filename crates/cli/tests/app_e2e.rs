use std::{
    env,
    fs::{self, read_to_string},
    path::Path,
    process::Command,
};

use eyre::Result;
use itertools::Itertools;
use tempfile::tempdir;

#[test]
fn test_cli_app_e2e() -> Result<()> {
    let temp_dir = tempdir()?;
    run_cmd("cargo", &["install", "--path", ".", "--force", "--locked"])?;
    let exe_path = "tests/programs/fibonacci/target/openvm/release/openvm-cli-example-test.vmexe";
    let temp_pk = temp_dir.path().join("app.pk");
    let temp_vk = temp_dir.path().join("app.vk");
    let temp_proof = temp_dir.path().join("fibonacci.app.proof");

    run_cmd(
        "cargo",
        &[
            "openvm",
            "build",
            "--manifest-path",
            "tests/programs/fibonacci/Cargo.toml",
            "--config",
            "tests/programs/fibonacci/openvm.toml",
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "keygen",
            "--config",
            "tests/programs/fibonacci/openvm.toml",
            "--output-dir",
            temp_dir.path().to_str().unwrap(),
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "run",
            "--exe",
            exe_path,
            "--config",
            "tests/programs/fibonacci/openvm.toml",
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "prove",
            "app",
            "--app-pk",
            temp_pk.to_str().unwrap(),
            "--exe",
            exe_path,
            "--proof",
            temp_proof.to_str().unwrap(),
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "verify",
            "app",
            "--app-vk",
            temp_vk.to_str().unwrap(),
            "--proof",
            temp_proof.to_str().unwrap(),
        ],
    )?;

    Ok(())
}

#[test]
fn test_cli_app_e2e_simplified() -> Result<()> {
    run_cmd("cargo", &["install", "--path", ".", "--force", "--locked"])?;
    run_cmd(
        "cargo",
        &[
            "openvm",
            "keygen",
            "--manifest-path",
            "tests/programs/multi/Cargo.toml",
        ],
    )?;
    run_cmd(
        "cargo",
        &[
            "openvm",
            "prove",
            "app",
            "--manifest-path",
            "tests/programs/multi/Cargo.toml",
            "--example",
            "fibonacci",
        ],
    )?;
    run_cmd(
        "cargo",
        &[
            "openvm",
            "verify",
            "app",
            "--manifest-path",
            "tests/programs/multi/Cargo.toml",
        ],
    )?;
    Ok(())
}

#[test]
fn test_cli_init_build() -> Result<()> {
    let temp_dir = tempdir()?;
    let temp_path = temp_dir.path();
    let config_path = temp_path.join("openvm.toml");
    let manifest_path = temp_path.join("Cargo.toml");
    run_cmd("cargo", &["install", "--path", ".", "--force", "--locked"])?;

    // Cargo will not respect patches if run within a workspace
    run_cmd(
        "cargo",
        &[
            "openvm",
            "init",
            temp_path.to_str().unwrap(),
            "--name",
            "cli-package",
        ],
    )?;
    if matches!(env::var("USE_LOCAL_OPENVM"), Ok(x) if x == "1") {
        replace_with_local_openvm(&manifest_path)?;
    }

    run_cmd(
        "cargo",
        &[
            "openvm",
            "build",
            "--config",
            config_path.to_str().unwrap(),
            "--manifest-path",
            manifest_path.to_str().unwrap(),
        ],
    )?;

    Ok(())
}

fn run_cmd(program: &str, args: &[&str]) -> Result<()> {
    let package_dir = env::current_dir()?;
    let prefix = "[test cli e2e]";
    println!(
        "{prefix} Running command: {} {} {} ...",
        program, args[0], args[1]
    );
    let mut cmd = Command::new(program);
    cmd.args(args);
    cmd.current_dir(package_dir);
    let output = cmd.output()?;
    println!("{prefix} Finished!");
    println!("{prefix} stdout:");
    println!("{}", std::str::from_utf8(&output.stdout).unwrap());
    println!("{prefix} stderr:");
    println!("{}", std::str::from_utf8(&output.stderr).unwrap());
    if !output.status.success() {
        return Err(eyre::eyre!("Command failed with status: {}", output.status));
    }
    Ok(())
}

fn replace_with_local_openvm(file_path: impl AsRef<Path>) -> Result<()> {
    const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
    let openvm_path = Path::new(MANIFEST_DIR)
        .parent()
        .unwrap()
        .join("toolchain")
        .join("openvm");
    let content = read_to_string(&file_path)?;
    let lines = content.lines().collect::<Vec<_>>();
    let new_content = lines
        .iter()
        .map(|line| {
            if line.starts_with("openvm = { git = \"https://github.com/openvm-org/openvm.git\"") {
                format!(
                    r#"openvm = {{ path = "{}", features = ["std"] }}"#,
                    openvm_path.display()
                )
            } else {
                line.to_string()
            }
        })
        .join("\n");

    fs::write(file_path, new_content)?;

    Ok(())
}
