use std::{fs, time::Instant};

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::keygen::MultiStarkKeygenBuilder;
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageConfig,
};
use bin_common::utils::io::{create_prefix, write_bytes};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_util::log2_strict_usize;

use super::common::{string_to_comp, CommonCommands, PAGE_BUS_INDEX, RANGE_BUS_INDEX};

#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[command(flatten)]
    pub common: CommonCommands,
}

impl KeygenCommand {
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        let cmp = string_to_comp(self.common.predicate);
        let output_folder = self.common.output_folder;

        let start = Instant::now();
        let idx_len = config.page.index_bytes / 2;
        let data_len = config.page.data_bytes / 2;
        let page_width = 1 + idx_len + data_len;
        let page_height = config.page.height;
        let idx_limb_bits = config.page.bits_per_fe;
        let idx_decomp = log2_strict_usize(page_height);
        let range_max = 1 << idx_decomp;

        let page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            range_max,
            idx_limb_bits,
            idx_decomp,
            cmp.clone(),
        );

        let engine = config::baby_bear_poseidon2::default_engine(idx_decomp);
        let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
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

        if !self.common.silent {
            println!("Keygen completed in {:?}", start.elapsed());
            println!("Partial proving key written to {}", pk_path);
            println!("Partial verifying key written to {}", vk_path);
        }
        Ok(())
    }
}
