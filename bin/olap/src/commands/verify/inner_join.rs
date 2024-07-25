use std::marker::PhantomData;

use afs_page::inner_join::controller::FKInnerJoinController;
use afs_stark_backend::{
    config::PcsProverData, keygen::types::MultiStarkProvingKey, prover::types::Proof,
};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::read_from_path;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::AfsOperation;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::de::DeserializeOwned;

use crate::{commands::CommonCommands, operations::inner_join::inner_join_setup};

#[derive(Debug, Parser)]
pub struct VerifyInnerJoinCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> VerifyInnerJoinCommand<SC, E>
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
        proof_path: Option<String>,
    ) -> Result<()> {
        let (
            t1_format,
            t2_format,
            inner_join_buses,
            inner_join_op,
            _page_left,
            _page_right,
            _height,
            range_chip_idx_decomp,
        ) = inner_join_setup(config, common, op);

        let inner_join_controller = FKInnerJoinController::new(
            inner_join_buses,
            t1_format,
            t2_format,
            range_chip_idx_decomp,
        );

        let prefix = config.generate_filename();
        let encoded_pk = read_from_path(keys_folder.clone() + "/" + &prefix + ".pk").unwrap();
        let pk: MultiStarkProvingKey<SC> = bincode::deserialize(&encoded_pk).unwrap();
        let vk = pk.vk();

        // Get proof from disk
        let table_id_full = inner_join_op.table_id_left.to_string();
        let default_proof_path = format!("bin/olap/tmp/cache/{}.proof.bin", table_id_full);
        let proof_path = proof_path.unwrap_or(default_proof_path);
        let encoded_proof = read_from_path(proof_path).unwrap();
        let proof: Proof<SC> = bincode::deserialize(&encoded_proof).unwrap();

        // Verify proof
        inner_join_controller.verify(engine, vk, proof).unwrap();

        Ok(())
    }
}
