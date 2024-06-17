use std::{
    fs::{remove_file, File},
    io::{copy, BufReader, BufWriter},
};

use afs_chips::{execution_air::ExecutionAir, page_rw_checker::page_controller::PageController};
use afs_stark_backend::{keygen::types::MultiStarkPartialVerifyingKey, prover::types::Proof};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
    page_config::{PageConfig, PageMode},
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_util::log2_strict_usize;

use crate::commands::read_from_path;

use super::create_prefix;

/// `afs verify` command
/// Uses information from config.toml to verify a proof using the verifying key in `output-folder`
/// as */prove.bin.
#[derive(Debug, Parser)]
pub struct VerifyCommand {
    #[arg(
        long = "proof-file",
        short = 'f',
        help = "The path to the proof file",
        required = true
    )]
    pub proof_file: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "DB file input (default: new empty DB)",
        required = true
    )]
    pub init_db_file_path: String,

    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains keys",
        required = false,
        default_value = "keys"
    )]
    pub keys_folder: String,
}

impl VerifyCommand {
    /// Execute the `verify` command
    pub fn execute(&self, config: &PageConfig) -> Result<()> {
        let prefix = create_prefix(&config);
        match config.page.mode {
            PageMode::ReadWrite => self.execute_rw(config, prefix)?,
            PageMode::ReadOnly => panic!(),
        }
        Ok(())
    }

    pub fn execute_rw(&self, config: &PageConfig, prefix: String) -> Result<()> {
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let height = config.page.height;

        assert!(height > 0);
        let page_bus_index = 0;
        let range_bus_index = 2;
        let ops_bus_index = 3;

        let checker_trace_degree = config.page.max_rw_ops as usize * 4;

        let idx_limb_bits = config.page.bits_per_fe as usize;

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(height))
            .max(8);

        let idx_decomp = 8;
        println!("Verifying proof file: {}", self.proof_file);
        // verify::verify_ops(&self.proof_file).await?;
        let encoded_vk =
            read_from_path(self.keys_folder.clone() + "/" + &prefix + ".partial.vk").unwrap();
        let partial_vk: MultiStarkPartialVerifyingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_vk).unwrap();

        let encoded_proof = read_from_path(self.proof_file.clone()).unwrap();
        let proof: Proof<BabyBearPoseidon2Config> = bincode::deserialize(&encoded_proof).unwrap();
        let page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let verifier = engine.verifier();
        let pis = vec![vec![]; partial_vk.per_air.len()];
        let mut challenger = engine.new_challenger();
        let result = verifier.verify(
            &mut challenger,
            partial_vk,
            vec![
                &page_controller.init_chip,
                &page_controller.final_chip,
                &page_controller.offline_checker,
                &page_controller.range_checker.air,
                &ops_sender,
            ],
            proof,
            &pis,
        );
        if result.is_err() {
            println!("Verification Unsuccessful");
        } else {
            println!("Verification Succeeded!");
            println!("Updates Committed");
            {
                let init_file = File::open(self.init_db_file_path.clone()).unwrap();
                let new_file = File::open(self.init_db_file_path.clone() + ".0").unwrap();
                let mut reader = BufReader::new(new_file);
                let mut writer = BufWriter::new(init_file);
                copy(&mut reader, &mut writer).unwrap();
            }
            remove_file(self.init_db_file_path.clone() + ".0").unwrap();
        }
        Ok(())
    }
}
