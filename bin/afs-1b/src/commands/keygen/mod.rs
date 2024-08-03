use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use afs_page::{common::page::Page, multitier_page_rw_checker::page_controller::PageController};
use afs_stark_backend::{
    config::{Com, PcsProverData},
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    engine::StarkEngine,
    page_config::{MultitierPageConfig, PageMode},
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::{PrimeField, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

use super::{create_prefix, BABYBEAR_COMMITMENT_LEN};
use crate::commands::{get_ops_sender, get_page_controller};

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
        Com<SC>: Into<[Val<SC>; BABYBEAR_COMMITMENT_LEN]>,
    {
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => Self::execute_rw(engine, config, output_folder, prefix)?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Generated keys in {:?}", duration);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_rw<SC: StarkGenericConfig, E>(
        engine: &E,
        config: &MultitierPageConfig,
        output_folder: String,
        prefix: String,
    ) -> Result<()>
    where
        E: StarkEngine<SC>,
        Val<SC>: PrimeField + PrimeField64,
        PcsProverData<SC>: Serialize + DeserializeOwned,
        Com<SC>: Into<[Val<SC>; BABYBEAR_COMMITMENT_LEN]>,
    {
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let leaf_height = config.page.leaf_height;
        let internal_height = config.page.internal_height;
        let page_controller: PageController<SC, BABYBEAR_COMMITMENT_LEN> =
            get_page_controller(config, idx_len, data_len);
        let ops_sender = get_ops_sender(idx_len, data_len);
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

        page_controller.set_up_keygen_builder(&mut keygen_builder, &ops_sender);

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
