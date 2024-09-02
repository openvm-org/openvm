use std::{
    fs::{remove_file, File},
    io::{copy, BufReader, BufWriter},
    marker::PhantomData,
    time::Instant,
};

use afs_page::page_rw_checker::page_controller::PageController;
use afs_stark_backend::{keygen::types::MultiStarkVerifyingKey, prover::types::Proof};
use ax_sdk::{
    engine::StarkEngine,
    page_config::{PageConfig, PageMode},
};
use bin_common::utils::io::read_from_path;
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::RANGE_CHECK_BITS;

/// `afs verify` command
/// Uses information from config.toml to verify a proof using the verifying key in `output-folder`
/// as */prove.bin.
#[derive(Debug, Parser)]
pub struct VerifyCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
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

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> VerifyCommand<SC, E>
where
    Val<SC>: PrimeField64,
{
    /// Execute the `verify` command
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        proof_file: String,
        init_db_file_path: String,
        keys_folder: String,
    ) -> Result<()> {
        let start = Instant::now();
        let prefix = config.generate_filename();
        match config.page.mode {
            PageMode::ReadWrite => Self::execute_rw(
                config,
                engine,
                prefix,
                proof_file,
                init_db_file_path,
                keys_folder,
            )?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Verified table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_rw(
        config: &PageConfig,
        engine: &E,
        prefix: String,
        proof_file: String,
        init_db_file_path: String,
        keys_folder: String,
    ) -> Result<()> {
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let height = config.page.height;

        assert!(height > 0);
        let page_bus_index = 0;
        let range_bus_index = 1;
        let ops_bus_index = 2;

        let idx_limb_bits = config.page.bits_per_fe;
        let idx_decomp = RANGE_CHECK_BITS;
        println!("Verifying proof file: {}", proof_file);

        let encoded_vk = read_from_path(keys_folder.clone() + "/" + &prefix + ".vk").unwrap();
        let vk: MultiStarkVerifyingKey<SC> = bincode::deserialize(&encoded_vk).unwrap();

        let encoded_proof = read_from_path(proof_file.clone()).unwrap();
        let proof: Proof<SC> = bincode::deserialize(&encoded_proof).unwrap();
        let page_controller: PageController<SC> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let result = page_controller.verify(engine, vk, proof);
        if result.is_err() {
            println!("Verification Unsuccessful");
        } else {
            println!("Verification Succeeded!");
            println!("Updates Committed");
            {
                let init_file = File::create(init_db_file_path.clone()).unwrap();
                let new_file = File::open(init_db_file_path.clone() + ".0").unwrap();
                let mut reader = BufReader::new(new_file);
                let mut writer = BufWriter::new(init_file);
                copy(&mut reader, &mut writer).unwrap();
            }
            remove_file(init_db_file_path.clone() + ".0").unwrap();
        }
        Ok(())
    }
}
