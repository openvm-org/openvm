use std::{
    fs::{create_dir, remove_dir_all},
    io::{BufRead, BufReader, Error},
    path::Path,
    process::Stdio,
};

const DIR_NAME: &str = "_edsl-kernel";
const SRC_DIR_NAME: &str = "_edsl-kernel/src";

pub fn execute_edsl(source: String, config: String) -> Result<String, Error> {
    let dir = Path::new(DIR_NAME);
    let _ = remove_dir_all(dir);

    create_dir(dir)?;
    let src_dir = Path::new(SRC_DIR_NAME);
    create_dir(src_dir)?;

    let src_file = Path::new(SRC_DIR_NAME).join("main.rs");
    std::fs::write(src_file, source)?;
    let config_file = Path::new(DIR_NAME).join("../../../../Cargo.toml");
    std::fs::write(config_file, config)?;

    let mut cmd = std::process::Command::new("cargo");

    cmd.arg("run").current_dir(dir);

    let mut child = cmd
        .stderr(Stdio::piped())
        .env("CARGO_TERM_COLOR", "always")
        .spawn()
        .expect("cargo build failed");
    let stderr = child.stderr.take().unwrap();
    for line in BufReader::new(stderr).lines() {
        println!("edsl kernel macro: {}", line.unwrap());
    }

    let output = cmd.output();

    let output = output?;

    let result = std::str::from_utf8(&output.stdout)
        .expect("failed to convert output to string")
        .to_string();

    //let err = std::str::from_utf8(&output.stderr).expect("failed to convert stderr to string").to_string();

    //let working_dir = std::env::current_dir().expect("failed to get current dir");

    remove_dir_all(dir)?;

    Ok(result)
}
