use afs_1b::cli::run;
use ax_sdk::page_config::MultitierPageConfig;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = MultitierPageConfig::read_config_file("config-1b.toml");
    run(&config);
}
