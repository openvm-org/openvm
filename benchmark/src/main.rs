use std::{
    fs::{create_dir_all, File},
    io,
    sync::Mutex,
};

use benchmark::{cli::Cli, utils::tracing::setup_benchmark_tracing, TMP_FOLDER, TMP_TRACING_LOG};

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = create_dir_all(TMP_FOLDER);
    let guard = setup_benchmark_tracing();
    Cli::run();
}
