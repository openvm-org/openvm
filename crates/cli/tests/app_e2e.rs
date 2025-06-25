use std::{env, process::Command, sync::OnceLock};

use eyre::Result;
use tempfile::tempdir;

fn install_cli() {
    static FORCE_INSTALL: OnceLock<bool> = OnceLock::new();
    FORCE_INSTALL.get_or_init(|| {
        if !matches!(env::var("SKIP_INSTALL"), Ok(x) if !x.is_empty()) {
            run_cmd("cargo", &["install", "--path", ".", "--force"]).unwrap();
        }
        true
    });
}

#[test]
fn test_cli_app_e2e() -> Result<()> {
    let temp_dir = tempdir()?;
    install_cli();
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
    install_cli();
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
    install_cli();

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
