<<<<<<< HEAD
use afs::cli::Cli;
=======
use afs::cli::run;
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b
use afs_test_utils::{config::setup_tracing, page_config::PageConfig};

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    let config = PageConfig::read_config_file("config.toml");
    setup_tracing();
<<<<<<< HEAD
    let _cli = Cli::run(&config);
=======
    run(&config);
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b
}
