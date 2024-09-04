use std::{fs, marker::PhantomData};

use afs_page::inner_join::controller::FKInnerJoinController;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkProvingKey,
    prover::trace::{ProverTraceData, TraceCommitmentBuilder},
};
use ax_sdk::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::{read_from_path, write_bytes};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::AfsOperation;
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::de::DeserializeOwned;

use crate::{commands::CommonCommands, operations::inner_join::inner_join_setup};

#[derive(Debug, Parser)]
pub struct ProveInnerJoinCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> ProveInnerJoinCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        op: AfsOperation,
        keys_folder: String,
        cache_folder: String,
    ) -> Result<()> {
        let (
            t1_format,
            t2_format,
            inner_join_buses,
            inner_join_op,
            page_left,
            page_right,
            height,
            range_chip_idx_decomp,
        ) = inner_join_setup(config, common, op);

        let mut inner_join_controller = FKInnerJoinController::new(
            inner_join_buses,
            t1_format,
            t2_format,
            range_chip_idx_decomp,
        );

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        // Deserialize the proving key
        let prefix = config.generate_filename();
        let encoded_pk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let pk: MultiStarkProvingKey<SC> = bincode::deserialize(&encoded_pk).unwrap();

        // Get the trace data from file
        let table_id_full = inner_join_op.table_id_left.to_string();
        let prover_trace_data_encoded =
            read_from_path(cache_folder.clone() + "/" + &table_id_full + ".cache.bin").unwrap();
        let (page1_input_pdata, page2_input_pdata, _page_output_pdata): (
            ProverTraceData<SC>,
            ProverTraceData<SC>,
            ProverTraceData<SC>,
        ) = bincode::deserialize(&prover_trace_data_encoded).unwrap();

        // Generate and encode the trace data
        let prover_trace_data = inner_join_controller.load_tables(
            &page_left,
            &page_right,
            Some(page1_input_pdata),
            Some(page2_input_pdata),
            None,
            2 * height,
            &mut trace_builder.committer,
        );

        // Generate a proof and write to file
        let proof = inner_join_controller.prove(engine, &pk, &mut trace_builder, prover_trace_data);
        let encoded_proof = bincode::serialize(&proof).unwrap();
        let proof_path = cache_folder.clone() + "/" + &table_id_full + ".proof.bin";
        let _ = fs::create_dir_all(&cache_folder);
        write_bytes(&encoded_proof, proof_path).unwrap();

        Ok(())
    }
}
