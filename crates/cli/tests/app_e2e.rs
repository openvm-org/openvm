use std::{env, process::Command};

use eyre::Result;
use tempfile::tempdir;

#[test]
fn test_cli_app_e2e() -> Result<()> {
    let temp_dir = tempdir()?;
    run_cmd("cargo", &["install", "--path", ".", "--force"])?;
    let temp_exe = temp_dir.path().join("example.vmexe");
    let temp_pk = temp_dir.path().join("example.pk");
    let temp_vk = temp_dir.path().join("example.vk");
    let temp_proof = temp_dir.path().join("example.apppf");

    run_cmd(
        "cargo",
        &[
            "openvm",
            "build",
            "--manifest-dir",
            "../sdk/example",
            "--transpile",
            "--transpiler-config",
            "example/app_config.toml",
            "--transpile-to",
            temp_exe.to_str().unwrap(),
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "keygen",
            "--config",
            "example/app_config.toml",
            "--output",
            temp_pk.to_str().unwrap(),
            "--vk-output",
            temp_vk.to_str().unwrap(),
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "run",
            "--exe",
            temp_exe.to_str().unwrap(),
            "--config",
            "example/app_config.toml",
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
            temp_exe.to_str().unwrap(),
            "--output",
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
fn test_cli_app_e2e_default_paths() -> Result<()> {
    let temp_dir = tempdir()?;
    run_cmd("cargo", &["install", "--path", ".", "--force"])?;
    let temp_exe = temp_dir.path().join("example.vmexe");

    run_cmd(
        "cargo",
        &[
            "openvm",
            "build",
            "--manifest-dir",
            "../sdk/example",
            "--transpile",
            "--transpiler-config",
            "example/app_config.toml",
            "--transpile-to",
            temp_exe.to_str().unwrap(),
        ],
    )?;

    run_cmd(
        "cargo",
        &["openvm", "keygen", "--config", "example/app_config.toml"],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "run",
            "--exe",
            temp_exe.to_str().unwrap(),
            "--config",
            "example/app_config.toml",
        ],
    )?;

    run_cmd(
        "cargo",
        &[
            "openvm",
            "prove",
            "app",
            "--exe",
            temp_exe.to_str().unwrap(),
        ],
    )?;

    run_cmd("cargo", &["openvm", "verify", "app"])?;

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
