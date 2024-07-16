use std::{fs, marker::PhantomData};

use afs_chips::inner_join::controller::FKInnerJoinController;
use afs_stark_backend::config::PcsProverData;
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::write_bytes;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::AfsOperation;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use crate::{commands::CommonCommands, operations::inner_join::inner_join_setup};

#[derive(Debug, Parser)]
pub struct KeygenInnerJoinCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> KeygenInnerJoinCommand<SC, E>
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
            t1_format,
            t2_format,
            inner_join_buses,
            _inner_join_op,
            _page_left,
            _page_right,
            _height,
            range_chip_idx_decomp,
        ) = inner_join_setup(config, common, op);

        let inner_join_controller = FKInnerJoinController::new(
            inner_join_buses,
            t1_format,
            t2_format,
            range_chip_idx_decomp,
        );
        let mut keygen_builder = engine.keygen_builder();
        inner_join_controller.set_up_keygen_builder(&mut keygen_builder);
        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();

        let prefix = config.generate_filename();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        let pk_path = keys_folder.clone() + "/" + &prefix.clone() + ".partial.pk";
        let vk_path = keys_folder.clone() + "/" + &prefix.clone() + ".partial.vk";
        let _ = fs::create_dir_all(&keys_folder);
        write_bytes(&encoded_pk, pk_path).unwrap();
        write_bytes(&encoded_vk, vk_path).unwrap();

        Ok(())
    }
}
