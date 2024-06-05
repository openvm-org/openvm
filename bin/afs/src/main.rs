use afs::{
    cli, 
    commands::{
        cache::CacheCommand, keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand
    }, utils::Config
};
use clap::Parser;
// use commands::{cache::CacheCommand, prove::ProveCommand};

fn main() {
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    let _config = Config::read_config_file("config.toml");

    let cli = cli::Cli::parse();
    
    match &cli.command {
        cli::types::CliCommand::Keygen(keygen) => {
            // println!("!Caching page file: {}", cache.page_file);
            let cmd = KeygenCommand {
                output_folder: keygen.output_folder.clone(),
            };
            cmd.execute().unwrap();
        }
        cli::types::CliCommand::Cache(cache) => {
            // println!("!Caching page file: {}", cache.page_file);
            let cmd = CacheCommand {
                page_file: cache.page_file.clone(),
                output_file: cache.output_file.clone(),
            };
            cmd.execute().unwrap();
        }
        cli::types::CliCommand::Prove(prove) => {
            let cmd = ProveCommand {
                ops_file: prove.ops_file.clone(),
                output_file: prove.output_file.clone(),
            };
            cmd.execute().unwrap();
        }
        cli::types::CliCommand::Verify(verify) => {
            let cmd = VerifyCommand {
                proof_file: verify.proof_file.clone(),
            };
            cmd.execute().unwrap();
        }
    }
}
