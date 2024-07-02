use std::sync::Arc;

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::{
    keygen::types::MultiStarkPartialProvingKey,
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder},
        MultiTraceStarkProver,
    },
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::PageConfig,
};
use bin_common::utils::{
    io::{create_prefix, read_from_path, write_bytes},
    page::print_page_nowrap,
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb, utils::string_to_u16_vec};

use super::{common_setup, comp_value_to_string, CommonCommands, PAGE_BUS_INDEX, RANGE_BUS_INDEX};

#[derive(Debug, Parser)]
pub struct ProveCommand {
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
        long = "db-file",
        short = 'd',
        help = "Path to the database file",
        required = true
    )]
    pub db_file_path: String,

    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains the proving and verifying keys",
        required = false,
        default_value = "bin/common/data/predicate"
    )]
    pub keys_folder: String,

    #[arg(
        long = "input-trace-file",
        short = 'i',
        help = "Input prover trace data file",
        required = true
    )]
    pub input_trace_file: String,

    #[arg(
        long = "output-trace-folder",
        short = 'u',
        help = "Folder to save output prover trace data file",
        required = false,
        default_value = "bin/common/data/predicate"
    )]
    pub output_trace_folder: String,

    #[command(flatten)]
    pub common: CommonCommands,
}

impl ProveCommand {
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        let table_id = self.table_id;
        let db_file_path = self.db_file_path;
        let output_folder = self.common.output_folder;

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
        ) = common_setup(config, self.common.predicate);
        let value = string_to_u16_vec(self.value, idx_len);

        // Get Page from db
        let mut db = MockDb::from_file(db_file_path.as_str());
        let interface = AfsInterface::new_with_table(table_id.clone(), &mut db);
        let table = interface.current_table().unwrap();

        // Handle prover trace data
        let input_trace_file = read_from_path(self.input_trace_file).unwrap();
        let input_trace_file: ProverTraceData<BabyBearPoseidon2Config> =
            bincode::deserialize(&input_trace_file).unwrap();

        // Get input page from trace data
        let page_input =
            table.to_page(config.page.index_bytes, config.page.data_bytes, page_height);

        if !self.common.silent {
            println!("Input page:");
            print_page_nowrap(&page_input);
        }

        let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            PAGE_BUS_INDEX,
            RANGE_BUS_INDEX,
            idx_len,
            data_len,
            range_max as u32,
            idx_limb_bits,
            idx_decomp,
            comp.clone(),
        );

        // Generate the output page
        let page_output =
            page_controller.gen_output(page_input.clone(), value.clone(), page_width, comp.clone());

        let engine = config::baby_bear_poseidon2::default_engine(idx_decomp);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let (input_prover_data, output_prover_data) = page_controller.load_page(
            page_input.clone(),
            page_output.clone(),
            Some(Arc::new(input_trace_file)),
            None,
            value.clone(),
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
            &mut trace_builder.committer,
        );

        // let output_trace = page_output.gen_trace::<BabyBear>();
        let output_trace_path = self.output_trace_folder.clone()
            + "/"
            + &table_id.clone()
            + comp_value_to_string(comp.clone(), value.clone()).as_str()
            + ".prover.cache.bin";
        let output_prover_data_ref = output_prover_data.as_ref();
        let encoded_output_trace_data: Vec<u8> =
            bincode::serialize(output_prover_data_ref).unwrap();
        write_bytes(&encoded_output_trace_data, output_trace_path).unwrap();

        // Load from disk and deserialize partial proving key
        let prefix = create_prefix(config);
        let encoded_pk =
            read_from_path(self.keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_pk).unwrap();

        // Prove
        let proof = page_controller.prove(
            &engine,
            &partial_pk,
            &mut trace_builder,
            input_prover_data,
            output_prover_data,
            value.clone(),
            idx_decomp,
        );

        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let proof_path =
            output_folder.clone() + "/" + &table_id.clone() + "-" + &prefix + ".prove.bin";
        write_bytes(&encoded_proof, proof_path.clone()).unwrap();

        if !self.common.silent {
            println!("Output page:");
            print_page_nowrap(&page_output);
            println!("Proving completed in {:?}", start.elapsed());
            println!("Proof written to {}", proof_path);
        }
        Ok(())
    }
}
