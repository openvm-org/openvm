use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use afs_chips::{execution_air::ExecutionAir, page_rw_checker::page_controller::PageController};
use afs_stark_backend::keygen::MultiStarkKeygenBuilder;
use afs_test_utils::page_config::PageConfig;
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageMode,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_util::log2_strict_usize;

use super::create_prefix;

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.partial.pk and *.partial.vk.
#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "keys"
    )]
    pub output_folder: String,
}

impl KeygenCommand {
    /// Execute the `keygen` command
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => self.execute_rw(
                (config.page.index_bytes + 1) / 2,
                (config.page.data_bytes + 1) / 2,
                config.page.max_rw_ops,
                config.page.height,
                config.page.bits_per_fe,
                prefix,
            )?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Generated keys in {:?}", duration);
        Ok(())
    }

    fn execute_rw(
        self,
        idx_len: usize,
        data_len: usize,
        max_ops: usize,
        height: usize,
        limb_bits: usize,
        prefix: String,
    ) -> Result<()> {
        let page_bus_index = 0;
        let range_bus_index = 1;
        let ops_bus_index = 2;

        let page_height = height;
        let checker_trace_degree = max_ops * 4;
        let idx_limb_bits = limb_bits;

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(page_height))
            .max(8);

        let idx_decomp = 8;

        let page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);

        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

        page_controller.set_up_keygen_builder(
            &mut keygen_builder,
            page_height,
            checker_trace_degree,
            &ops_sender,
            max_ops,
        );

        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        let pk_path = self.output_folder.clone() + "/" + &prefix.clone() + ".partial.pk";
        let vk_path = self.output_folder.clone() + "/" + &prefix.clone() + ".partial.vk";
        fs::create_dir_all(self.output_folder).unwrap();
        write_bytes(&encoded_pk, pk_path).unwrap();
        write_bytes(&encoded_vk, vk_path).unwrap();
        Ok(())
    }
}

fn write_bytes(bytes: &[u8], path: String) -> Result<()> {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}
