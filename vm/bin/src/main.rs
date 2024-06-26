use afs_test_utils::config::setup_tracing;
use cli::cli::Cli;
use stark_vm::vm::config::VmConfig;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = VmConfig::read_config_file("config.toml").unwrap();
    setup_tracing();
    let _cli = Cli::run(config);
}
