use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    sync::Arc,
    time::Instant,
};

use afs_page::{
    common::page::Page,
    execution_air::ExecutionAir,
    multitier_page_rw_checker::page_controller::{
        MyLessThanTupleParams, PageController, PageTreeParams,
    },
};
use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    config::PcsProverData,
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{config::baby_bear_poseidon2::BabyBearPoseidon2Config, page_config::PageMode};
use afs_test_utils::{
    engine::StarkEngine,
    page_config::{MultitierPageConfig, TreeParamsConfig},
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::{PrimeField, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

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
    pub fn execute<SC: StarkGenericConfig, E>(
        config: &MultitierPageConfig,
        engine: &E,
        output_folder: String,
    ) -> Result<()>
    where
        E: StarkEngine<SC>,
        Val<SC>: PrimeField + PrimeField64,
        PcsProverData<SC>: Serialize + DeserializeOwned,
    {
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => Self::execute_rw(
                engine,
                output_folder,
                (config.page.index_bytes + 1) / 2,
                (config.page.data_bytes + 1) / 2,
                config.page.leaf_height,
                config.page.internal_height,
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

    #[allow(clippy::too_many_arguments)]
    fn execute_rw<SC: StarkGenericConfig, E>(
        engine: &E,
        output_folder: String,
        idx_len: usize,
        data_len: usize,
        leaf_height: usize,
        internal_height: usize,
        _limb_bits: usize,
        tree_params: &TreeParamsConfig,
        prefix: String,
    ) -> Result<()>
    where
        E: StarkEngine<SC>,
        Val<SC>: PrimeField + PrimeField64,
        PcsProverData<SC>: Serialize + DeserializeOwned,
    {
        let data_bus_index = 0;
        let internal_data_bus_index = 1;
        let lt_bus_index = 2;

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
                    leaf_page_height: leaf_height,
                    internal_page_height: internal_height,
                },
                PageTreeParams {
                    path_bus_index: final_path_bus,
                    leaf_cap: tree_params.final_leaf_cap,
                    internal_cap: tree_params.final_internal_cap,
                    leaf_page_height: leaf_height,
                    internal_page_height: internal_height,
                },
                less_than_tuple_param,
                range_checker,
            );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let mut keygen_builder = MultiStarkKeygenBuilder::new(engine.config());

        let prover = MultiTraceStarkProver::new(engine.config());
        let trace_builder = TraceCommitmentBuilder::<SC>::new(prover.pcs());

        let blank_leaf = vec![vec![0; 1 + idx_len + data_len]; leaf_height];

        let blank_leaf = Page::from_2d_vec_consume(blank_leaf, idx_len, data_len);

        let mut blank_internal_row = vec![2];
        blank_internal_row.resize(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN, 0);
        let blank_internal = vec![blank_internal_row; internal_height];

        // literally use any leaf chip
        let blank_leaf_trace =
            page_controller.init_leaf_chips[0].generate_cached_trace_from_page(&blank_leaf);
        let blank_internal_trace =
            page_controller.init_internal_chips[0].generate_cached_trace(&blank_internal);
        let blank_leaf_prover_data = trace_builder.committer.commit(vec![blank_leaf_trace]);
        let blank_internal_prover_data = trace_builder.committer.commit(vec![blank_internal_trace]);

        fs::create_dir_all(output_folder.clone()).unwrap();

        let encoded_data = bincode::serialize(&blank_leaf_prover_data).unwrap();
        write_bytes(
            &encoded_data,
            output_folder.clone() + "/" + &prefix.clone() + ".blank_leaf.cache.bin",
        )
        .unwrap();

        let encoded_data = bincode::serialize(&blank_internal_prover_data).unwrap();
        write_bytes(
            &encoded_data,
            output_folder.clone() + "/" + &prefix.clone() + ".blank_internal.cache.bin",
        )
        .unwrap();

        let mut init_leaf_data_ptrs = vec![];

        let mut init_internal_data_ptrs = vec![];
        let mut init_internal_main_ptrs = vec![];

        let mut final_leaf_data_ptrs = vec![];
        let mut final_leaf_main_ptrs = vec![];

        let mut final_internal_data_ptrs = vec![];
        let mut final_internal_main_ptrs = vec![];

        for _ in 0..tree_params.init_leaf_cap {
            init_leaf_data_ptrs.push(keygen_builder.add_cached_main_matrix(1 + idx_len + data_len));
        }

        for _ in 0..tree_params.init_internal_cap {
            init_internal_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN),
            );
        }

        for _ in 0..tree_params.final_leaf_cap {
            final_leaf_data_ptrs
                .push(keygen_builder.add_cached_main_matrix(1 + idx_len + data_len));
        }

        for _ in 0..tree_params.final_internal_cap {
            final_internal_data_ptrs.push(
                keygen_builder.add_cached_main_matrix(2 + 2 * idx_len + BABYBEAR_COMMITMENT_LEN),
            );
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
                page_controller.final_leaf_chips[0].air_width() - 1 - idx_len - data_len,
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

        for (chip, ptr) in page_controller
            .init_leaf_chips
            .iter()
            .zip(init_leaf_data_ptrs.into_iter())
        {
            keygen_builder.add_partitioned_air(chip, BABYBEAR_COMMITMENT_LEN, vec![ptr]);
        }

        for i in 0..tree_params.init_internal_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.init_internal_chips[i],
                BABYBEAR_COMMITMENT_LEN,
                vec![init_internal_data_ptrs[i], init_internal_main_ptrs[i]],
            );
        }

        for i in 0..tree_params.final_leaf_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.final_leaf_chips[i],
                BABYBEAR_COMMITMENT_LEN,
                vec![final_leaf_data_ptrs[i], final_leaf_main_ptrs[i]],
            );
        }

        for i in 0..tree_params.final_internal_cap {
            keygen_builder.add_partitioned_air(
                &page_controller.final_internal_chips[i],
                BABYBEAR_COMMITMENT_LEN,
                vec![final_internal_data_ptrs[i], final_internal_main_ptrs[i]],
            );
        }

        keygen_builder.add_partitioned_air(&page_controller.offline_checker, 0, vec![ops_ptr]);

        keygen_builder.add_partitioned_air(
            &page_controller.init_root_signal,
            BABYBEAR_COMMITMENT_LEN,
            vec![init_root_ptr],
        );

        keygen_builder.add_partitioned_air(
            &page_controller.final_root_signal,
            BABYBEAR_COMMITMENT_LEN,
            vec![final_root_ptr],
        );

        keygen_builder.add_air(&page_controller.range_checker.air, 0);

        keygen_builder.add_air(&ops_sender, 0);

        let pk = keygen_builder.generate_pk();

        let vk = pk.vk();
        let (total_preprocessed, total_partitioned_main, total_after_challenge) =
            vk.total_air_width();
        let air_width = total_preprocessed + total_partitioned_main + total_after_challenge;
        info!("Keygen: total air width: {}", air_width);
        println!("Keygen: total air width: {}", air_width);
        let encoded_pk: Vec<u8> = bincode::serialize(&pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&vk)?;
        let pk_path = output_folder.clone() + "/" + &prefix.clone() + ".partial.pk";
        let vk_path = output_folder.clone() + "/" + &prefix.clone() + ".partial.vk";
        write_bytes(&encoded_pk, pk_path).unwrap();
        write_bytes(&encoded_vk, vk_path).unwrap();
        Ok(())
    }
}

fn write_bytes(bytes: &[u8], path: String) -> Result<()> {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes).unwrap();
    Ok(())
}
