use std::{fs, marker::PhantomData};

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::config::PcsProverData;
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::{create_prefix, write_bytes};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use super::{common_setup, CommonCommands, PAGE_BUS_INDEX, RANGE_BUS_INDEX};

#[derive(Debug, Parser)]
pub struct KeygenCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(flatten)]
    pub common: CommonCommands,

    #[clap(skip)]
    pub _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> KeygenCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    pub fn execute(config: &PageConfig, engine: &E, common: &CommonCommands) -> Result<()> {
        let output_folder = common.output_folder.clone();

        let (
            start,
            comp,
            idx_len,
            data_len,
            page_width,
            page_height,
            idx_limb_bits,
            idx_decomp,
            range_max,
        ) = common_setup(config, common.predicate.clone());

        let page_controller: PageController<SC> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            range_max as u32,
            idx_limb_bits,
            idx_decomp,
            comp,
        );

        let mut keygen_builder = engine.keygen_builder();
        page_controller.set_up_keygen_builder(
            &mut keygen_builder,
            page_width,
            page_height,
            idx_len,
            idx_decomp,
        );

        // Write the partial pk and vk to disk
        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        let prefix = create_prefix(config);
        let pk_path = output_folder.clone() + "/" + &prefix.clone() + ".partial.pk";
        let vk_path = output_folder.clone() + "/" + &prefix.clone() + ".partial.vk";
        fs::create_dir_all(output_folder).unwrap();
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
