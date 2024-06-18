use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use afs_chips::{
    execution_air::ExecutionAir, multitier_page_rw_checker::page_controller::PageController,
};
use afs_stark_backend::{keygen::MultiStarkKeygenBuilder, prover::MultiTraceStarkProver};
use afs_test_utils::page_config::{MultitierPageConfig, PageConfig, TreeParamsConfig};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageMode,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_util::log2_strict_usize;

use super::create_prefix;
use super::BABYBEAR_COMMITMENT_LEN;
use super::DECOMP_BITS;
use super::LIMB_BITS;

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
    pub fn execute(self, config: &MultitierPageConfig) -> Result<()> {
        let start = Instant::now();
        let prefix = create_prefix(&config);
        match config.page.mode {
            PageMode::ReadWrite => self.execute_rw(
                (config.page.index_bytes + 1) / 2 as usize,
                (config.page.data_bytes + 1) / 2 as usize,
                config.page.max_rw_ops as usize,
                config.page.height as usize,
                config.page.bits_per_fe,
                &config.tree,
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
        tree_params: &TreeParamsConfig,
        prefix: String,
    ) -> Result<()> {
        let data_bus_index = 0;
        let internal_data_bus_index = 1;
        let lt_bus_index = 2;

        let page_height = height;

        let trace_degree = max_ops * 4;

        let engine =
            config::baby_bear_poseidon2::default_engine(log_page_height.max(3 + log_num_ops));
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let init_path_bus = 3;
        let final_path_bus = 4;
        let ops_bus_index = 5;

        let less_than_tuple_param = MyLessThanTupleParams {
            limb_bits: LIMB_BITS,
            decomp: DECOMP_BITS,
        };

        let range_checker = Arc::new(RangeCheckerGateChip::new(lt_bus_index, 1 << DECOMP_BITS));

        let mut page_controller: PageController<BABYBEAR_COMMITMENT_LEN> =
            PageController::new::<BabyBearPoseidon2Config>(
                data_bus_index,
                internal_data_bus_index,
                ops_bus_index,
                lt_bus_index,
                idx_len,
                data_len,
                init_param.clone(),
                final_param.clone(),
                less_than_tuple_param,
                range_checker,
            );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

        let mut init_leaf_data_ptrs = vec![];
        let mut init_leaf_main_ptrs = vec![];

        let mut init_internal_data_ptrs = vec![];
        let mut init_internal_main_ptrs = vec![];

        let mut final_leaf_data_ptrs = vec![];
        let mut final_leaf_main_ptrs = vec![];

        let mut final_internal_data_ptrs = vec![];
        let mut final_internal_main_ptrs = vec![];

        for _ in 0..tree_params.init_leaf_cap {
            init_leaf_data_ptrs.push(keygen_builder.add_cached_main_matrix(2 + idx_len + data_len));
        }

        for _ in 0..tree_params.init_internal_cap {
            init_internal_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN),
            );
        }

        for _ in 0..tree_params.final_leaf_cap {
            final_leaf_data_ptrs
                .push(keygen_builder.add_cached_main_matrix(2 + idx_len + data_len));
        }

        for _ in 0..tree_params.final_internal_cap {
            final_internal_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN),
            );
        }

        for _ in 0..tree_params.init_leaf_cap {
            init_leaf_main_ptrs.push(keygen_builder.add_main_matrix(
                page_controller.init_leaf_chips[0].air_width() - 2 - idx_len - data_len,
            ));
        }

        for _ in 0..tree_params.init_internal_cap {
            init_internal_main_ptrs.push(keygen_builder.add_main_matrix(
                page_controller.init_internal_chips[0].air_width()
                    - 2
                    - 2 * idx_len
                    - BABYBEAR_COMMITMENT_LEN,
            ));
        }

        for _ in 0..tree_params.final_leaf_cap {
            final_leaf_main_ptrs.push(keygen_builder.add_main_matrix(
                page_controller.final_leaf_chips[0].air_width() - 2 - idx_len - data_len,
            ));
        }

        for _ in 0..tree_params.final_internal_cap {
            final_internal_main_ptrs.push(keygen_builder.add_main_matrix(
                page_controller.final_internal_chips[0].air_width()
                    - 2
                    - 2 * idx_len
                    - BABYBEAR_COMMITMENT_LEN,
            ));
        }

        let ops_ptr = keygen_builder.add_main_matrix(page_controller.offline_checker.air_width());

        let init_root_ptr =
            keygen_builder.add_main_matrix(page_controller.init_root_signal.air_width());
        let final_root_ptr =
            keygen_builder.add_main_matrix(page_controller.final_root_signal.air_width());

        for i in 0..init_param.leaf_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.init_leaf_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![init_leaf_data_ptrs[i], init_leaf_main_ptrs[i]],
            );
        }

        for i in 0..init_param.internal_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.init_internal_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![init_internal_data_ptrs[i], init_internal_main_ptrs[i]],
            );
        }

        for i in 0..final_param.leaf_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.final_leaf_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![final_leaf_data_ptrs[i], final_leaf_main_ptrs[i]],
            );
        }

        for i in 0..final_param.internal_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.final_internal_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![final_internal_data_ptrs[i], final_internal_main_ptrs[i]],
            );
        }

        keygen_builder.add_partitioned_air(
            &page_controller.offline_checker,
            trace_degree,
            0,
            vec![ops_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.init_root_signal,
            1,
            BABYBEAR_COMMITMENT_LEN,
            vec![init_root_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.final_root_signal,
            1,
            BABYBEAR_COMMITMENT_LEN,
            vec![final_root_ptr],
        );

        keygen_builder.add_air(&page_controller.range_checker.air, 1 << DECOMP_BITS, 0);

        keygen_builder.add_air(&ops_sender, num_ops, 0);

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
