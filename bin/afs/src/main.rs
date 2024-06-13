use afs::{
    cli::Cli,
    common::config::Config,
};

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = Config::read_config_file("config.toml");
    let _cli = Cli::run(&config);
}
