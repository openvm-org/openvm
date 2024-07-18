use std::marker::PhantomData;

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::{
    config::PcsProverData, keygen::types::MultiStarkPartialVerifyingKey, prover::types::Proof,
};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::read_from_path;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::types::AfsOperation,
    utils::{string_to_u16_vec, u16_vec_to_hex_string},
};
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::de::DeserializeOwned;

use crate::{
    commands::CommonCommands,
    operations::filter::{filter_setup, PAGE_BUS_INDEX, RANGE_BUS_INDEX},
    CACHE_FOLDER,
};

#[derive(Debug, Parser)]
pub struct VerifyFilterCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> VerifyFilterCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: DeserializeOwned,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        op: AfsOperation,
        keys_folder: String,
        cache_folder: Option<String>,
        proof_path: Option<String>,
    ) -> Result<()> {
        let (
            start,
            filter_op,
            idx_len,
            data_len,
            _page_width,
            _page_height,
            idx_limb_bits,
            idx_decomp,
            range_max,
        ) = filter_setup(config, op);

        let value = string_to_u16_vec(filter_op.value, idx_len);
        let table_id = filter_op.table_id.to_string();

        let page_controller: PageController<SC> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            range_max as u32,
            idx_limb_bits,
            idx_decomp,
            filter_op.predicate.clone(),
        );

        // Load from disk and deserialize partial verifying key
        let prefix = config.generate_filename();
        let encoded_vk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.vk").unwrap();
        let partial_vk: MultiStarkPartialVerifyingKey<SC> =
            bincode::deserialize(&encoded_vk).unwrap();

        // Get proof from disk
        let filter_info = format!(
            "{}{}",
            filter_op.predicate,
            u16_vec_to_hex_string(value.clone())
        );
        let cache_folder = cache_folder.unwrap_or(CACHE_FOLDER.to_string());
        let default_proof_path = format!("{}/{}-{}.proof.bin", cache_folder, table_id, filter_info);
        let proof_path = proof_path.unwrap_or(default_proof_path);
        let encoded_proof = read_from_path(proof_path).unwrap();
        let proof: Proof<SC> = bincode::deserialize(&encoded_proof).unwrap();

        // Verify proof
        page_controller
            .verify(engine, partial_vk, proof, value)
            .unwrap();

        if !common.silent {
            println!("Proof verified in {:?}", start.elapsed());
        }
        Ok(())
    }
}
