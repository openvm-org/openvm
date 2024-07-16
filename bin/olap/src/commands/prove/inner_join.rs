use std::{fs, marker::PhantomData};

use afs_chips::inner_join::controller::{FKInnerJoinController, IJTraces};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkPartialProvingKey,
    prover::trace::{ProverTraceData, TraceCommitmentBuilder},
};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
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
            _bits_per_fe,
            _degree,
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

        let prefix = config.generate_filename();
        let encoded_pk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<SC> =
            bincode::deserialize(&encoded_pk).unwrap();

        let table_id_full = inner_join_op.table_id_left.to_string();
        let prover_trace_data_encoded =
            read_from_path(cache_folder.clone() + "/" + &table_id_full + ".cache.bin").unwrap();
        let (prover_trace_data, inner_join_traces): (Vec<ProverTraceData<SC>>, IJTraces<Val<SC>>) =
            bincode::deserialize(&prover_trace_data_encoded).unwrap();

        inner_join_controller.generate_prover_traces(&page_left, &page_right, 2 * height);

        let proof = inner_join_controller.prove(
            engine,
            &partial_pk,
            &mut trace_builder,
            prover_trace_data,
            &inner_join_traces,
        );
        let encoded_proof = bincode::serialize(&proof).unwrap();
        let proof_path = cache_folder.clone() + "/" + &table_id_full + ".proof.bin";
        let _ = fs::create_dir_all(&cache_folder);
        write_bytes(&encoded_proof, proof_path).unwrap();

        Ok(())
    }
}
