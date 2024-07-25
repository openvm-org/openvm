use std::fs::create_dir_all;

use benchmark::{cli::Cli, utils::tracing::setup_benchmark_tracing, TMP_FOLDER};

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let _ = create_dir_all(TMP_FOLDER);
    setup_benchmark_tracing();
    Cli::run();
}
