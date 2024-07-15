use afs_test_utils::{config::setup_tracing, page_config::PageConfig};
use olap::cli::run;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = PageConfig::read_config_file("config.toml");
    setup_tracing();
    run(&config);
}
