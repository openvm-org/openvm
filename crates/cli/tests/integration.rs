use std::{env, process::Command, sync::OnceLock};

use eyre::Result;

fn install_cli() {
    static FORCE_INSTALL: OnceLock<bool> = OnceLock::new();
    FORCE_INSTALL.get_or_init(|| {
        if !matches!(env::var("SKIP_INSTALL"), Ok(x) if !x.is_empty()) {
            run_cmd("cargo", &["install", "--path", ".", "--force", "--locked"]).unwrap();
        }
        true
    });
}

fn build_fibonacci_once() -> Result<&'static str> {
    static BUILD_ONCE: OnceLock<Result<()>> = OnceLock::new();
    BUILD_ONCE
        .get_or_init(|| {
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
            )
        })
        .as_ref()
        .map(|_| "tests/programs/fibonacci/openvm/release/openvm-cli-example-test.vmexe")
        .map_err(|e| eyre::eyre!("Failed to build fibonacci: {}", e))
}

#[test]
fn test_cli_app_e2e() -> Result<()> {
    install_cli();
    let exe_path = build_fibonacci_once()?;
    run_script("cli_app_e2e.sh", &[("EXE_PATH", exe_path)])
}

#[test]
fn test_cli_app_e2e_simplified() -> Result<()> {
    install_cli();
    run_script("cli_app_e2e_simplified.sh", &[])
}

#[test]
fn test_cli_stark_e2e_simplified() -> Result<()> {
    install_cli();
    run_script("cli_stark_e2e_simplified.sh", &[])
}

#[test]
fn test_cli_init_build() -> Result<()> {
    install_cli();
    run_script("cli_init_build.sh", &[])
}

#[test]
fn test_cli_run_mode_pure_default() -> Result<()> {
    install_cli();
    let exe_path = build_fibonacci_once()?;
    run_script("cli_run_mode_pure_default.sh", &[("EXE_PATH", exe_path)])
}

#[test]
fn test_cli_run_segment() -> Result<()> {
    install_cli();
    let exe_path = build_fibonacci_once()?;
    run_script("cli_run_segment.sh", &[("EXE_PATH", exe_path)])
}

#[test]
fn test_cli_run_meter() -> Result<()> {
    install_cli();
    let exe_path = build_fibonacci_once()?;
    run_script("cli_run_meter.sh", &[("EXE_PATH", exe_path)])
}

fn run_cmd(program: &str, args: &[&str]) -> Result<()> {
    let package_dir = env::current_dir()?;
    let prefix = "[test cli e2e]";
    println!(
        "{prefix} Running command: {program} {} {} ...",
        args[0], args[1]
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

fn run_script(script_name: &str, envs: &[(&str, &str)]) -> Result<()> {
    let package_dir = env::current_dir()?;
    let script_path = package_dir.join("tests").join("scripts").join(script_name);
    let prefix = "[test cli e2e]";
    println!("{prefix} Running script: {}", script_path.display());
    let mut cmd = Command::new("bash");
    cmd.arg(script_path);
    cmd.current_dir(package_dir);
    cmd.envs(envs.iter().copied());
    let output = cmd.output()?;
    println!("{prefix} Finished!");
    println!("{prefix} stdout:");
    println!("{}", std::str::from_utf8(&output.stdout).unwrap());
    println!("{prefix} stderr:");
    println!("{}", std::str::from_utf8(&output.stderr).unwrap());
    if !output.status.success() {
        return Err(eyre::eyre!("Script failed with status: {}", output.status));
    }
    Ok(())
}
