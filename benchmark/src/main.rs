use afs_test_utils::config::setup_tracing;
use benchmark::cli::Cli;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    setup_tracing();
    Cli::run();
}
