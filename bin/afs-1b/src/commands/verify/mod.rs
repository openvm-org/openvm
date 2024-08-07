use std::{
    fs::{remove_file, File},
    io::{copy, BufReader, BufWriter},
    time::Instant,
};

use afs_page::multitier_page_rw_checker::page_controller::PageController;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkVerifyingKey,
    prover::types::Proof,
};
use afs_test_utils::{
    engine::StarkEngine,
    page_config::{MultitierPageConfig, PageMode},
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::{PrimeField, PrimeField32, PrimeField64};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use super::create_prefix;
use crate::commands::{
    get_ops_sender, get_page_controller, read_from_path, BABYBEAR_COMMITMENT_LEN,
};

/// `afs verify` command
/// Uses information from config.toml to verify a proof using the verifying key in `output-folder`
/// as */prove.bin.
#[derive(Debug, Parser)]
pub struct VerifyCommand {
    #[arg(long = "table-id", short = 't', help = "The table ID", required = true)]
    pub table_id: String,

    #[arg(
        long = "db-folder",
        short = 'd',
        help = "Mock DB folder (default: new empty DB)",
        required = false,
        default_value = "multitier_mockdb"
    )]
    pub db_folder: String,

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
    pub fn execute<SC: StarkGenericConfig, E>(
        config: &MultitierPageConfig,
        engine: &E,
        table_id: String,
        db_folder: String,
        keys_folder: String,
    ) -> Result<()>
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
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => {
                Self::execute_rw(config, engine, table_id, db_folder, keys_folder, prefix)?
            }
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Verified table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_rw<SC: StarkGenericConfig, E>(
        config: &MultitierPageConfig,
        engine: &E,
        table_id: String,
        db_folder: String,
        keys_folder: String,
        prefix: String,
    ) -> Result<()>
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
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let proof_path = db_folder.clone() + "/" + &table_id + ".prove.bin";
        let original_root = db_folder.clone() + "/root/" + &table_id;
        println!("Verifying proof file: {}", proof_path);
        // verify::verify_ops(&proof_file).await?;
        let encoded_vk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.vk").unwrap();
        let vk: MultiStarkVerifyingKey<SC> = bincode::deserialize(&encoded_vk).unwrap();

        let encoded_proof = read_from_path(proof_path).unwrap();
        let proof: Proof<SC> = bincode::deserialize(&encoded_proof).unwrap();
        let pis_path = db_folder.clone() + "/" + &table_id + ".pi.bin";
        let encoded_pis = read_from_path(pis_path).unwrap();
        let pis: Vec<Vec<Val<SC>>> = bincode::deserialize(&encoded_pis).unwrap();

        let page_controller: PageController<SC, BABYBEAR_COMMITMENT_LEN> =
            get_page_controller(config, idx_len, data_len);
        let ops_sender = get_ops_sender(idx_len, data_len);
        let result = page_controller.verify(engine, &vk, &proof, &pis, &ops_sender);
        if result.is_err() {
            println!("Verification Unsuccessful");
        } else {
            println!("Verification Succeeded!");
            println!("Updates Committed");
            {
                let init_file = File::create(original_root.clone()).unwrap();
                let new_file = File::open(original_root.clone() + ".0").unwrap();
                let mut reader = BufReader::new(new_file);
                let mut writer = BufWriter::new(init_file);
                copy(&mut reader, &mut writer).unwrap();
            }
            remove_file(original_root.clone() + ".0").unwrap();
        }
        Ok(())
    }
}
