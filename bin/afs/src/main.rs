use afs::cli::run;
use ax_sdk::{config::setup_tracing, page_config::PageConfig};

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = PageConfig::read_config_file("config.toml");
    setup_tracing();
    run(&config);
}
