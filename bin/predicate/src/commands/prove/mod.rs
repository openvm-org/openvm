use std::{sync::Arc, time::Instant};

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
use p3_util::log2_strict_usize;

use super::common::{string_to_comp, CommonCommands, PAGE_BUS_INDEX, RANGE_BUS_INDEX};

#[derive(Debug, Parser)]
pub struct ProveCommand {
    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains the proving and verifying keys",
        required = false,
        default_value = "bin/common/data/predicate"
    )]
    pub keys_folder: String,

    #[arg(
        long = "input-trace-data",
        short = 'i',
        help = "The input prover trace data",
        required = false
    )]
    pub input_trace_data: Option<String>,

    #[arg(
        long = "output-trace-data",
        short = 'u',
        help = "The output prover trace data",
        required = false
    )]
    pub output_trace_data: Option<String>,

    #[command(flatten)]
    pub common: CommonCommands,
}

impl ProveCommand {
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        let cmp = string_to_comp(self.common.predicate);
        let value = self.common.value;
        let table_id = self.common.table_id;
        let db_file_path = self.common.db_file_path;
        let output_folder = self.common.output_folder;

        let start = Instant::now();
        let idx_len = config.page.index_bytes / 2;
        let data_len = config.page.data_bytes / 2;
        let page_width = 1 + idx_len + data_len;
        let page_height = config.page.height;
        let idx_limb_bits = config.page.bits_per_fe;
        let idx_decomp = log2_strict_usize(page_height);
        let range_max = 1 << idx_decomp;
        let value = string_to_u16_vec(value, idx_len);

        // Get Page from db
        let mut db = MockDb::from_file(db_file_path.as_str());
        let interface = AfsInterface::new_with_table(table_id.clone(), &mut db);
        let table = interface.current_table().unwrap();
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
            range_max,
            idx_limb_bits,
            idx_decomp,
            cmp.clone(),
        );

        // Generate the output page
        let page_output =
            page_controller.gen_output(page_input.clone(), value.clone(), page_width, cmp);

        let engine = config::baby_bear_poseidon2::default_engine(idx_decomp);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        // Handle optional prover data
        let input_trace_data = if self.input_trace_data.is_some() {
            let trace_data_path = self.input_trace_data.unwrap();
            let trace_data = read_from_path(trace_data_path).unwrap();
            let trace_data: ProverTraceData<BabyBearPoseidon2Config> =
                bincode::deserialize(&trace_data).unwrap();
            Some(Arc::new(trace_data))
        } else {
            None
        };
        let output_trace_data = if self.output_trace_data.is_some() {
            let trace_data_path = self.output_trace_data.unwrap();
            let trace_data = read_from_path(trace_data_path).unwrap();
            let trace_data: ProverTraceData<BabyBearPoseidon2Config> =
                bincode::deserialize(&trace_data).unwrap();
            Some(Arc::new(trace_data))
        } else {
            None
        };

        let (input_prover_data, output_prover_data) = page_controller.load_page(
            page_input.clone(),
            page_output.clone(),
            input_trace_data,
            output_trace_data,
            value.clone(),
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
            &mut trace_builder.committer,
        );

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
