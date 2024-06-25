use afs_test_utils::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Config, setup_tracing},
    page_config::PageConfig,
};
use olap::cli::Cli;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = PageConfig::read_config_file("config.toml");
    setup_tracing();
    let _cli = Cli::run::<BabyBearPoseidon2Config>(&config);
}
