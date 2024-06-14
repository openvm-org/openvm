use std::{
    fs::{self, File},
    io::{BufWriter, Write},
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
        // WIP: Wait for ReadWrite chip in https://github.com/axiom-crypto/afs-prototype/pull/45
        let prefix = create_prefix(&config);
        match config.page.mode {
            PageMode::ReadWrite => self.execute_rw(
                (config.page.index_bytes + 1) / 2 as usize,
                (config.page.data_bytes + 1) / 2 as usize,
                config.page.max_rw_ops as usize,
                config.page.height as usize,
                config.page.bits_per_fe,
                prefix,
            )?,
            PageMode::ReadOnly => panic!(),
        }
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
        let checker_final_bus_index = 1;
        let range_bus_index = 2;
        let ops_bus_index = 3;

        let page_height = height;
        let page_width = 1 + idx_len + data_len;

        let checker_trace_degree = max_ops * 4;

        let idx_limb_bits = limb_bits;

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(page_height))
            .max(8);

        let idx_decomp = 8;

        let page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            page_bus_index,
            checker_final_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);

        // i put a dummy max value here - to be changed
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

        let init_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
        let final_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
        let final_page_aux_ptr =
            keygen_builder.add_main_matrix(page_controller.final_chip.aux_width());
        let offline_checker_ptr =
            keygen_builder.add_main_matrix(page_controller.offline_checker.air_width());
        let range_checker_ptr =
            keygen_builder.add_main_matrix(page_controller.range_checker.air_width());
        let ops_sender_ptr = keygen_builder.add_main_matrix(ops_sender.air_width());

        keygen_builder.add_partitioned_air(
            &page_controller.init_chip,
            page_height,
            0,
            vec![init_page_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.final_chip,
            page_height,
            0,
            vec![final_page_ptr, final_page_aux_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.offline_checker,
            checker_trace_degree,
            0,
            vec![offline_checker_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.range_checker.air,
            1 << idx_decomp,
            0,
            vec![range_checker_ptr],
        );

        keygen_builder.add_partitioned_air(&ops_sender, max_ops, 0, vec![ops_sender_ptr]);

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

fn write_bytes(bytes: &Vec<u8>, path: String) -> Result<()> {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write(bytes).unwrap();
    Ok(())
}
