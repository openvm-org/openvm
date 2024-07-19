use std::{fs, marker::PhantomData, time::Instant};

use afs_stark_backend::{config::PcsProverData, prover::trace::TraceCommitmentBuilder};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::write_bytes;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::types::AfsOperation, afs_interface::AfsInterface, mock_db::MockDb,
};
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use crate::{commands::CommonCommands, operations::filter::filter_setup};

#[derive(Debug, Parser)]
pub struct CacheFilterCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> CacheFilterCommand<SC, E>
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
            _start,
            filter_op,
            _idx_len,
            _data_len,
            _page_width,
            _page_height,
            _idx_limb_bits,
            _idx_decomp,
            _range_max,
        ) = filter_setup(config, op);

        let start = Instant::now();
        let mut db = MockDb::from_file(&common.db_path);
        let table_id = filter_op.table_id.to_string();
        let height = config.page.height;
        assert!(height > 0);

        let mut interface =
            AfsInterface::new(config.page.index_bytes, config.page.data_bytes, &mut db);
        let page = interface.get_table(table_id.clone()).unwrap().to_page(
            config.page.index_bytes,
            config.page.data_bytes,
            height,
        );

        let trace = page.gen_trace::<Val<SC>>();
        let prover = engine.prover();
        let trace_builder = TraceCommitmentBuilder::<SC>::new(prover.pcs());
        let prover_trace_data = trace_builder.committer.commit(vec![trace]);
        let encoded_data = bincode::serialize(&prover_trace_data).unwrap();
        let path = cache_folder.clone() + "/" + &table_id + ".cache.bin";
        let _ = fs::create_dir_all(&cache_folder);
        write_bytes(&encoded_data, path).unwrap();

        let duration = start.elapsed();
        println!("Cached table {} in {:?}", table_id, duration);

        Ok(())
    }
}
