use afs::cli::Cli;
use afs_test_utils::{config::setup_tracing, engine::engine_from_params, page_config::PageConfig};
use p3_util::log2_strict_usize;

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = PageConfig::read_config_file("config.toml");
    let pcs_log_degree = log2_strict_usize(config.page.height);
    let engine_type = config.stark_engine.engine;
    let fri_params = config.fri_params;
    let engine = engine_from_params(engine_type, fri_params, pcs_log_degree).as_ref();
    setup_tracing();
    let _cli = Cli::run(&config, &engine);
}
