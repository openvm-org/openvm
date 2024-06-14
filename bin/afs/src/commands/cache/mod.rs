use afs_chips::page_rw_checker::page::PageAir;
use afs_stark_backend::prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageConfig,
};
use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb};
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
        // prove::prove_ops(&self.ops_file).await?;
        let mut db = MockDb::from_file(&self.db_file_path);
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let height = config.page.height;
        let mut interface = AfsInterface::<U256, U256>::new(&mut db);
        let page = interface
            .get_table(self.table_id.clone())
            .unwrap()
            .to_page(height);

        assert!(height > 0);
        let page_bus_index = 0;

        let checker_trace_degree = config.page.max_rw_ops as usize * 4;

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(height))
            .max(8);

        let page_chip = PageAir::new(page_bus_index, idx_len, data_len);
        let trace = page_chip.generate_trace::<BabyBearPoseidon2Config>(page);
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let trace_builder = TraceCommitmentBuilder::<BabyBearPoseidon2Config>::new(prover.pcs());
        let trace_prover_data = trace_builder.committer.commit(vec![trace]);
        // dummy value for now
        let encoded_data = bincode::serialize(&page_bus_index).unwrap();
        let path = self.output_folder.clone() + "/" + &self.table_id + ".cache.bin";
        write_bytes(&encoded_data, path).unwrap();
        // WIP: wait for PR #45: https://github.com/axiom-crypto/afs-prototype/pull/45
        Ok(())
    }

    // pub fn read_page_file(&self) -> Result<Vec<Vec<u32>>> {
    //     let page_file = std::fs::read(&self.page_file)?;
    //     let page_file: Vec<Vec<u32>> = serde_json::from_slice(&page_file)?;
    //     Ok(page_file)
    // }

    // pub fn write_output_file(&self, output: Vec<u8>) -> Result<()> {
    //     std::fs::write(&self.output_file, output)?;
    //     Ok(())
    // }
}
