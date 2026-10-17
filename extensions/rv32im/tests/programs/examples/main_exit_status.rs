fn main() -> Result<std::process::ExitCode, &'static str> {
    match openvm::io::read::<u8>() {
        0 => Err("guest failed"),
        code => Ok(std::process::ExitCode::from(code - 1)),
    }
}
