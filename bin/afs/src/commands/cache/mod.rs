use std::{marker::PhantomData, time::Instant};

use afs_page::common::page::Page;
use afs_stark_backend::{config::PcsProverData, prover::trace::TraceCommitmentBuilder};
use ax_sdk::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::write_bytes;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_interface::{utils::string_to_table_id, AfsInterface},
    mock_db::MockDb,
};
use p3_field::PrimeField;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

#[cfg(test)]
pub mod tests;

/// `afs cache` command
#[derive(Debug, Parser)]
pub struct CacheCommand<SC: StarkGenericConfig, E: StarkEngine<SC> + ?Sized> {
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
        long = "output-folder",
        short = 'o',
        help = "The folder to output the cached traces to",
        required = false,
        default_value = "cache"
    )]
    pub output_folder: String,

    #[clap(skip)]
    pub _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + ?Sized> CacheCommand<SC, E>
where
    Val<SC>: PrimeField,
    PcsProverData<SC>: Serialize,
{
    /// Execute the `cache` command
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        table_id: String,
        db_file_path: String,
        output_folder: String,
    ) -> Result<()> {
        println!("Caching table {} from {}", table_id, db_file_path);

        let start = Instant::now();
        let mut db = MockDb::from_file(&db_file_path);
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
        let table_id_full = string_to_table_id(table_id.clone()).to_string();
        let path = output_folder.clone() + "/" + &table_id_full + ".cache.bin";
        write_bytes(&encoded_data, path).unwrap();

        let duration = start.elapsed();
        println!("Cached table {} in {:?}", table_id, duration);

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
