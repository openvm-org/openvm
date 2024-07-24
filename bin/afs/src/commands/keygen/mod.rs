use std::fs;
use std::{marker::PhantomData, time::Instant};

use afs_chips::{execution_air::ExecutionAir, page_rw_checker::page_controller::PageController};
use afs_stark_backend::{config::PcsProverData, keygen::MultiStarkKeygenBuilder};
use afs_test_utils::page_config::PageMode;
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::write_bytes;
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;
use tracing::info;

use crate::RANGE_CHECK_BITS;

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.pk and *.vk.
#[derive(Debug, Parser)]
pub struct KeygenCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "keys"
    )]
    pub output_folder: String,

    #[clap(skip)]
    pub _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> KeygenCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    /// Execute the `keygen` command
    pub fn execute(config: &PageConfig, engine: &E, output_folder: String) -> Result<()> {
        let start = Instant::now();
        let prefix = config.generate_filename();
        match config.page.mode {
            PageMode::ReadWrite => KeygenCommand::execute_rw(
                engine,
                (config.page.index_bytes + 1) / 2,
                (config.page.data_bytes + 1) / 2,
                config.page.bits_per_fe,
                prefix,
                output_folder,
            )?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Generated keys in {:?}", duration);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_rw(
        engine: &E,
        idx_len: usize,
        data_len: usize,
        limb_bits: usize,
        prefix: String,
        output_folder: String,
    ) -> Result<()> {
        let page_bus_index = 0;
        let range_bus_index = 1;
        let ops_bus_index = 2;

        let idx_limb_bits = limb_bits;

        let idx_decomp = RANGE_CHECK_BITS;

        let page_controller: PageController<SC> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);

        let mut keygen_builder: MultiStarkKeygenBuilder<SC> = engine.keygen_builder();

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
        let pk_path = output_folder.clone() + "/" + &prefix.clone() + ".pk";
        let vk_path = output_folder.clone() + "/" + &prefix.clone() + ".vk";
        let _ = fs::create_dir_all(&output_folder);
        write_bytes(&encoded_pk, pk_path).unwrap();
        write_bytes(&encoded_vk, vk_path).unwrap();
        Ok(())
    }
}
