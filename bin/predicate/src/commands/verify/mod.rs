use std::marker::PhantomData;

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::{
    config::PcsProverData, keygen::types::MultiStarkPartialVerifyingKey, prover::types::Proof,
};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::io::{create_prefix, read_from_path};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{afs_interface::utils::string_to_table_id, utils::string_to_u16_vec};
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use super::{common_setup, CommonCommands, PAGE_BUS_INDEX, RANGE_BUS_INDEX};

#[derive(Debug, Parser)]
pub struct VerifyCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[arg(
        long = "value",
        short = 'v',
        help = "Value to prove the predicate against",
        required = true
    )]
    pub value: String,

    #[arg(
        long = "table-id",
        short = 't',
        help = "Table id to run the predicate on",
        required = true
    )]
    pub table_id: String,

    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains the proving and verifying keys",
        required = false,
        default_value = "bin/common/data/predicate"
    )]
    pub keys_folder: String,

    #[command(flatten)]
    pub common: CommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> VerifyCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        value: String,
        table_id: String,
        keys_folder: String,
    ) -> Result<()> {
        // Get full-length table_id
        let table_id_full = string_to_table_id(table_id).to_string();
        let output_folder = common.output_folder.clone();

        let (
            start,
            comp,
            idx_len,
            data_len,
            _page_width,
            _page_height,
            idx_limb_bits,
            idx_decomp,
            range_max,
        ) = common_setup(config, common.predicate.clone());
        let value = string_to_u16_vec(value, idx_len);

        let page_controller: PageController<SC> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            range_max as u32,
            idx_limb_bits,
            idx_decomp,
            comp.clone(),
        );

        // Load from disk and deserialize partial verifying key
        let prefix = create_prefix(config);
        let encoded_vk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.vk").unwrap();
        let partial_vk: MultiStarkPartialVerifyingKey<SC> =
            bincode::deserialize(&encoded_vk).unwrap();

        // Get proof
        let prefix = create_prefix(config);
        let encoded_proof = read_from_path(
            output_folder.clone() + "/" + &table_id_full + "-" + &prefix + ".prove.bin",
        )
        .unwrap();

        let proof: Proof<SC> = bincode::deserialize(&encoded_proof).unwrap();

        page_controller
            .verify(engine, partial_vk, proof, value)
            .unwrap();

        if !common.silent {
            println!("Proof verified in {:?}", start.elapsed());
        }

        Ok(())
    }
}
