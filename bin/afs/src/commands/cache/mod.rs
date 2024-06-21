use std::time::Instant;

use afs_chips::common::page::Page;
use afs_stark_backend::prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageConfig,
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb};
use p3_baby_bear::BabyBear;
use p3_util::log2_strict_usize;

use crate::commands::write_bytes;

#[cfg(test)]
pub mod tests;

/// `afs cache` command
#[derive(Debug, Parser)]
pub struct CacheCommand {
    #[arg(long = "table-id", short = 't', help = "The table ID", required = true)]
    pub table_id: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "Mock DB file input",
        required = true
    )]
    pub db_file_path: String,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "The folder to output the cached traces to",
        required = false,
        default_value = "cache"
    )]
    pub output_folder: String,
}

impl CacheCommand {
    /// Execute the `cache` command
    pub fn execute(&self, config: &PageConfig) -> Result<()> {
        println!("Caching table {} from {}", self.table_id, self.db_file_path);

        let start = Instant::now();
        let mut db = MockDb::from_file(&self.db_file_path);
        let height = config.page.height;
        let mut interface =
            AfsInterface::new(config.page.index_bytes, config.page.data_bytes, &mut db);
        let page = interface.get_table(self.table_id.clone()).unwrap().to_page(
            config.page.index_bytes,
            config.page.data_bytes,
            height,
        );

        assert!(height > 0);

        let checker_trace_degree = config.page.max_rw_ops * 4;

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(height))
            .max(8);

        let trace = page.gen_trace::<BabyBear>();
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let trace_builder = TraceCommitmentBuilder::<BabyBearPoseidon2Config>::new(prover.pcs());
        let trace_prover_data = trace_builder.committer.commit(vec![trace]);
        let encoded_data = bincode::serialize(&trace_prover_data).unwrap();
        let path = self.output_folder.clone() + "/" + &self.table_id + ".cache.bin";
        write_bytes(&encoded_data, path).unwrap();

        let duration = start.elapsed();
        println!("Cached table {} in {:?}", self.table_id, duration);

        Ok(())
    }

    pub fn read_page_file(&self) -> Result<Page> {
        let path = self.output_folder.clone() + "/" + &self.table_id + ".cache.bin";
        let page_file = std::fs::read(path)?;
        let page_file: Page = serde_json::from_slice(&page_file)?;
        Ok(page_file)
    }

    pub fn write_output_file(&self, output: Vec<u8>) -> Result<()> {
        let path = self.output_folder.clone() + "/" + &self.table_id + ".cache.bin";
        std::fs::write(path, output)?;
        Ok(())
    }
}
