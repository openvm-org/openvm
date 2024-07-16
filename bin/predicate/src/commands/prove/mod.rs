use std::{marker::PhantomData, sync::Arc};

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkPartialProvingKey,
    prover::trace::{ProverTraceData, TraceCommitmentBuilder},
};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::{
    io::{create_prefix, read_from_path, write_bytes},
    page::print_page_nowrap,
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_interface::{utils::string_to_table_id, AfsInterface},
    mock_db::MockDb,
    utils::string_to_u16_vec,
};
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use super::{common_setup, comp_value_to_string, CommonCommands, PAGE_BUS_INDEX, RANGE_BUS_INDEX};

#[derive(Debug, Parser)]
pub struct ProveCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
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

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> ProveCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        value: String,
        table_id: String,
        db_file_path: String,
        keys_folder: String,
        input_trace_file: String,
        output_trace_folder: String,
    ) -> Result<()> {
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
        let value = string_to_u16_vec(value, idx_len);

        // Get Page from db
        let mut db = MockDb::from_file(db_file_path.as_str());
        let interface = AfsInterface::new_with_table(table_id.clone(), &mut db);
        let table = interface.current_table().unwrap();

        // Handle prover trace data
        let input_trace_file = read_from_path(input_trace_file).unwrap();
        let input_trace_file: ProverTraceData<SC> =
            bincode::deserialize(&input_trace_file).unwrap();

        // Get input page from trace data
        let page_input =
            table.to_page(config.page.index_bytes, config.page.data_bytes, page_height);

        if !common.silent {
            println!("Input page:");
            print_page_nowrap(&page_input);
        }

        let mut page_controller: PageController<SC> = PageController::new(
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

        let prover = engine.prover();
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

        let output_trace_path = output_trace_folder.clone()
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
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<SC> =
            bincode::deserialize(&encoded_pk).unwrap();

        // Prove
        let proof = page_controller.prove(
            engine,
            &partial_pk,
            &mut trace_builder,
            input_prover_data,
            output_prover_data,
            value.clone(),
            idx_decomp,
        );

        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let table_id_full = string_to_table_id(table_id.clone()).to_string();
        let proof_path =
            output_folder.clone() + "/" + &table_id_full + "-" + &prefix + ".prove.bin";
        write_bytes(&encoded_proof, proof_path.clone()).unwrap();

        if !common.silent {
            println!("Output page:");
            print_page_nowrap(&page_output);
            println!("Proving completed in {:?}", start.elapsed());
            println!("Proof written to {}", proof_path);
        }
        Ok(())
    }
}
