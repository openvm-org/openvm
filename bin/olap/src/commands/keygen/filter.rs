use std::{fs, marker::PhantomData};

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::config::PcsProverData;
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::write_bytes;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::AfsOperation;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;
use tracing::info;

use crate::{
    commands::CommonCommands,
    operations::filter::{filter_setup, PAGE_BUS_INDEX, RANGE_BUS_INDEX},
};

#[derive(Debug, Parser)]
pub struct KeygenFilterCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> KeygenFilterCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        op: AfsOperation,
        keys_folder: String,
    ) -> Result<()> {
        let (
            start,
            filter_op,
            idx_len,
            data_len,
            page_width,
            _page_height,
            idx_limb_bits,
            idx_decomp,
            range_max,
        ) = filter_setup(config, op);

        let page_controller: PageController<SC> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            range_max as u32,
            idx_limb_bits,
            idx_decomp,
            filter_op.predicate,
        );
        let mut keygen_builder = engine.keygen_builder();
        page_controller.set_up_keygen_builder(&mut keygen_builder, page_width, idx_len);

        // Write the partial pk and vk to disk
        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let (total_preprocessed, total_partitioned_main, total_after_challenge) =
            partial_vk.total_air_width();
        let air_width = total_preprocessed + total_partitioned_main + total_after_challenge;
        info!("Keygen: total air width: {}", air_width);
        println!("Keygen: total air width: {}", air_width);

        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        let prefix = config.generate_filename();
        let pk_path = keys_folder.clone() + "/" + &prefix.clone() + ".partial.pk";
        let vk_path = keys_folder.clone() + "/" + &prefix.clone() + ".partial.vk";
        let _ = fs::create_dir_all(keys_folder);
        write_bytes(&encoded_pk, pk_path.clone()).unwrap();
        write_bytes(&encoded_vk, vk_path.clone()).unwrap();

        if !common.silent {
            println!("Keygen completed in {:?}", start.elapsed());
            println!("Partial proving key written to {}", pk_path);
            println!("Partial verifying key written to {}", vk_path);
        }
        Ok(())
    }
}
