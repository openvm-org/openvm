use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    sync::Arc,
    time::Instant,
};

use afs_chips::{
    common::page::Page,
    execution_air::ExecutionAir,
    multitier_page_rw_checker::page_controller::{
        MyLessThanTupleParams, PageController, PageTreeParams,
    },
    range_gate::RangeCheckerGateChip,
};
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::page_config::{MultitierPageConfig, TreeParamsConfig};
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
        _limb_bits: usize,
        tree_params: &TreeParamsConfig,
        prefix: String,
    ) -> Result<()> {
        let data_bus_index = 0;
        let internal_data_bus_index = 1;
        let lt_bus_index = 2;

        let page_height = height;

        let trace_degree = max_ops * 4;

        let log_page_height = log2_strict_usize(height);
        let log_trace_degree = 2 + log2_strict_usize(max_ops);

        let engine = config::baby_bear_poseidon2::default_engine(
            log_page_height.max(DECOMP_BITS).max(log_trace_degree),
        );

        let init_path_bus = 3;
        let final_path_bus = 4;
        let ops_bus_index = 5;

        let less_than_tuple_param = MyLessThanTupleParams {
            limb_bits: LIMB_BITS,
            decomp: DECOMP_BITS,
        };

        let range_checker = Arc::new(RangeCheckerGateChip::new(lt_bus_index, 1 << DECOMP_BITS));

        let page_controller: PageController<BABYBEAR_COMMITMENT_LEN> =
            PageController::new::<BabyBearPoseidon2Config>(
                data_bus_index,
                internal_data_bus_index,
                ops_bus_index,
                lt_bus_index,
                idx_len,
                data_len,
                PageTreeParams {
                    path_bus_index: init_path_bus,
                    leaf_cap: tree_params.init_leaf_cap,
                    internal_cap: tree_params.init_internal_cap,
                    leaf_page_height: height,
                    internal_page_height: height,
                },
                PageTreeParams {
                    path_bus_index: final_path_bus,
                    leaf_cap: tree_params.final_leaf_cap,
                    internal_cap: tree_params.final_internal_cap,
                    leaf_page_height: height,
                    internal_page_height: height,
                },
                less_than_tuple_param,
                range_checker,
            );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

        let engine = config::baby_bear_poseidon2::default_engine(
            log_page_height.max(DECOMP_BITS).max(log_trace_degree),
        );
        let prover = MultiTraceStarkProver::new(&engine.config);
        let trace_builder = TraceCommitmentBuilder::<BabyBearPoseidon2Config>::new(prover.pcs());
        let mut blank_leaf_row = vec![1];
        blank_leaf_row.resize(2 + idx_len + data_len, 0);
        let blank_leaf = vec![blank_leaf_row.clone(); height];

        let blank_internal = vec![vec![0; 2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN]; height];

        // literally use any leaf chip
        let blank_leaf_trace = page_controller.init_leaf_chips[0]
            .generate_cached_trace(Page::from_2d_vec(&blank_leaf, idx_len, data_len));
        let blank_internal_trace =
            page_controller.init_internal_chips[0].generate_cached_trace(blank_internal);
        let blank_leaf_prover_data = trace_builder.committer.commit(vec![blank_leaf_trace]);
        let blank_internal_prover_data = trace_builder.committer.commit(vec![blank_internal_trace]);

        fs::create_dir_all(self.output_folder.clone()).unwrap();

        let encoded_data = bincode::serialize(&blank_leaf_prover_data).unwrap();
        write_bytes(
            &encoded_data,
            self.output_folder.clone() + "/" + &prefix.clone() + ".blank_leaf.cache.bin",
        )
        .unwrap();

        let encoded_data = bincode::serialize(&blank_internal_prover_data).unwrap();
        write_bytes(
            &encoded_data,
            self.output_folder.clone() + "/" + &prefix.clone() + ".blank_internal.cache.bin",
        )
        .unwrap();

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

        for i in 0..tree_params.init_leaf_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.init_leaf_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![init_leaf_data_ptrs[i], init_leaf_main_ptrs[i]],
            );
        }

        for i in 0..tree_params.init_internal_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.init_internal_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![init_internal_data_ptrs[i], init_internal_main_ptrs[i]],
            );
        }

        for i in 0..tree_params.final_leaf_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.final_leaf_chips[i],
                page_height,
                BABYBEAR_COMMITMENT_LEN,
                vec![final_leaf_data_ptrs[i], final_leaf_main_ptrs[i]],
            );
        }

        for i in 0..tree_params.final_internal_cap {
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

        keygen_builder.add_air(&ops_sender, max_ops, 0);

        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        let pk_path = self.output_folder.clone() + "/" + &prefix.clone() + ".partial.pk";
        let vk_path = self.output_folder.clone() + "/" + &prefix.clone() + ".partial.vk";
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
