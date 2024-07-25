use crate::commands::keygen::KeygenCommand;
use crate::commands::prove::ProveCommand;
use crate::commands::verify::VerifyCommand;
use crate::commands::{keygen, mock, prove, verify, BABYBEAR_COMMITMENT_LEN, DECOMP_BITS};
use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::config::baby_bear_poseidon2::{
    engine_from_perm, random_perm, BabyBearPoseidon2Engine,
};
use afs_test_utils::config::EngineType;
use afs_test_utils::engine::StarkEngine;
use afs_test_utils::page_config::MultitierPageConfig;
use clap::Parser;
use clap::Subcommand;
use p3_field::{PrimeField, PrimeField32, PrimeField64};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS CLI")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(name = "mock", about = "Mock functions")]
    /// Mock functions
    Mock(mock::MockCommand),

    #[command(name = "keygen", about = "Generate partial proving and verifying keys")]
    /// Generate partial proving and verifying keys
    Keygen(keygen::KeygenCommand),

    #[command(name = "prove", about = "Generates a multi-STARK proof")]
    /// Generates a multi-STARK proof
    Prove(prove::ProveCommand),

    #[command(name = "verify", about = "Verifies a multi-STARK proof")]
    /// Verifies a multi-STARK proof
    Verify(verify::VerifyCommand),
}

impl Cli {
    pub fn run_with_engine<SC: StarkGenericConfig, E>(config: &MultitierPageConfig, engine: &E)
    where
        E: StarkEngine<SC>,
        Val<SC>: PrimeField + PrimeField64 + PrimeField32,
        Com<SC>: Into<[Val<SC>; BABYBEAR_COMMITMENT_LEN]>,
        PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
        PcsProof<SC>: Send + Sync,
        Domain<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Pcs: Sync,
        SC::Challenge: Send + Sync,
    {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Mock(mock) => {
                mock.execute(config).unwrap();
            }
            CliCommand::Keygen(keygen) => {
                let output_folder = keygen.output_folder.clone();
                KeygenCommand::execute(config, engine, output_folder).unwrap();
            }
            CliCommand::Prove(prove) => {
                let afi_file_path = prove.afi_file_path.clone();
                let db_file_path = prove.db_folder.clone();
                let keys_folder = prove.keys_folder.clone();
                let silent = prove.silent;
                ProveCommand::execute(
                    config,
                    engine,
                    afi_file_path,
                    db_file_path,
                    keys_folder,
                    silent,
                )
                .unwrap();
            }
            CliCommand::Verify(verify) => {
                let table_id = verify.table_id.clone();
                let db_folder = verify.db_folder.clone();
                let keys_folder = verify.keys_folder.clone();
                VerifyCommand::execute(config, engine, table_id, db_folder, keys_folder).unwrap();
            }
        }
    }
}

pub fn run(config: &MultitierPageConfig) {
    let checker_trace_degree = config.page.max_rw_ops * 4;
    let pcs_log_degree = log2_strict_usize(checker_trace_degree)
        .max(log2_strict_usize(config.page.leaf_height))
        .max(DECOMP_BITS);
    let fri_params = config.fri_params;
    let engine_type = config.stark_engine.engine;
    match engine_type {
        EngineType::BabyBearBlake3 => {
            // let engine: BabyBearBlake3Engine =
            //     engine_from_byte_hash(Blake3, pcs_log_degree, fri_params);
            // Cli::run_with_engine(config, &engine)
            panic!()
        }
        EngineType::BabyBearKeccak => {
            // let engine: BabyBearKeccakEngine =
            //     engine_from_byte_hash(Keccak256Hash, pcs_log_degree, fri_params);
            // Cli::run_with_engine(config, &engine)
            panic!()
        }
        EngineType::BabyBearPoseidon2 => {
            let perm = random_perm();
            let engine: BabyBearPoseidon2Engine =
                engine_from_perm(perm, pcs_log_degree, fri_params);
            Cli::run_with_engine(config, &engine)
        }
        _ => panic!(),
    }
}
