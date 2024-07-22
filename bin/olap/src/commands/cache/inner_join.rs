use std::{fs, marker::PhantomData};

use afs_chips::inner_join::controller::FKInnerJoinController;
use afs_stark_backend::{config::PcsProverData, prover::trace::TraceCommitmentBuilder};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::write_bytes;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::AfsOperation;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use crate::{commands::CommonCommands, operations::inner_join::inner_join_setup};

#[derive(Debug, Parser)]
pub struct CacheInnerJoinCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> CacheInnerJoinCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        op: AfsOperation,
        cache_folder: String,
    ) -> Result<()> {
        let (
            t1_format,
            t2_format,
            inner_join_buses,
            inner_join_op,
            page_left,
            page_right,
            _height,
            range_chip_idx_decomp,
        ) = inner_join_setup(config, common, op);

        let mut inner_join_controller = FKInnerJoinController::<SC>::new(
            inner_join_buses,
            t1_format,
            t2_format,
            range_chip_idx_decomp,
        );

        let (t1_main_trace, t2_main_trace, output_main_trace) =
            inner_join_controller.io_main_traces(&page_left, &page_right);

        let prover = engine.prover();
        let trace_builder = TraceCommitmentBuilder::<SC>::new(prover.pcs());

        let trace_committer = trace_builder.committer;
        let t1_commit = trace_committer.commit(vec![t1_main_trace.clone()]);
        let t2_commit = trace_committer.commit(vec![t2_main_trace.clone()]);
        let output_commit = trace_committer.commit(vec![output_main_trace.clone()]);

        let all_commits = (t1_commit, t2_commit, output_commit);
        let encoded_data = bincode::serialize(&all_commits).unwrap();

        // Save the traces data
        let table_id_full = inner_join_op.table_id_left.to_string();
        let path = cache_folder.clone() + "/" + &table_id_full + ".cache.bin";
        let _ = fs::create_dir_all(&cache_folder);
        write_bytes(&encoded_data, path).unwrap();

        Ok(())
    }
}
