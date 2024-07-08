use std::{fs::File, sync::Mutex};

use benchmark::{cli::Cli, TMP_TRACING_LOG};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::fmt::format::FmtSpan;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let tmp_log = File::create(TMP_TRACING_LOG.as_str()).unwrap();
    tracing_subscriber::fmt()
        .with_env_filter("benchmark=info,afs=info")
        .with_span_events(FmtSpan::CLOSE)
        .with_max_level(LevelFilter::INFO)
        .with_writer(Mutex::new(tmp_log))
        .with_ansi(false)
        .init();
    Cli::run();
}

/*
TTD:
x Use config folder
- Get benchmarks
- Display table
- Write CSV
*/
