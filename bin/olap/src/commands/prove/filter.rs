use std::{fs, marker::PhantomData, sync::Arc};

use afs_page::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkProvingKey,
    prover::trace::{ProverTraceData, TraceCommitmentBuilder},
};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use bin_common::utils::{
    io::{read_from_path, write_bytes},
    page::print_page_nowrap,
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::types::AfsOperation,
    afs_interface::AfsInterface,
    mock_db::MockDb,
    utils::{string_to_u16_vec, u16_vec_to_hex_string},
};
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    commands::CommonCommands,
    operations::filter::{comp_value_to_string, filter_setup, PAGE_BUS_INDEX, RANGE_BUS_INDEX},
};

#[derive(Debug, Parser)]
pub struct ProveFilterCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> ProveFilterCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        op: AfsOperation,
        keys_folder: String,
        cache_folder: String,
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

        let value = string_to_u16_vec(filter_op.value, idx_len);
        let table_id = filter_op.table_id.to_string();

        // Get Page from db
        let mut db = MockDb::from_file(common.db_path.as_str());
        let interface = AfsInterface::new_with_table(table_id.clone(), &mut db);
        let table = interface.current_table().unwrap();

        // Handle prover trace data
        let prover_trace_data_encoded =
            read_from_path(cache_folder.clone() + "/" + &table_id + ".cache.bin").unwrap();
        let input_trace_file: ProverTraceData<SC> =
            bincode::deserialize(&prover_trace_data_encoded).unwrap();

        // Get input page from trace data
        let page_input = table.to_page(
            config.page.index_bytes,
            config.page.data_bytes,
            config.page.height,
        );

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
            filter_op.predicate.clone(),
        );

        // Generate the output page
        let page_output = page_controller.gen_output(
            page_input.clone(),
            value.clone(),
            page_width,
            filter_op.predicate.clone(),
        );

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

        let output_trace_path = cache_folder.clone()
            + "/"
            + &table_id.clone()
            + comp_value_to_string(filter_op.predicate.clone(), value.clone()).as_str()
            + ".proof.bin";
        let output_prover_data_ref = output_prover_data.as_ref();
        let encoded_output_trace_data: Vec<u8> =
            bincode::serialize(output_prover_data_ref).unwrap();
        write_bytes(&encoded_output_trace_data, output_trace_path).unwrap();

        // Load from disk and deserialize partial proving key
        let prefix = config.generate_filename();
        let encoded_pk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let pk: MultiStarkProvingKey<SC> = bincode::deserialize(&encoded_pk).unwrap();

        // Prove
        let proof = page_controller.prove(
            engine,
            &pk,
            &mut trace_builder,
            input_prover_data,
            output_prover_data,
            value.clone(),
            idx_decomp,
        );

        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let filter_info = format!(
            "{}{}",
            filter_op.predicate,
            u16_vec_to_hex_string(value.clone())
        );
        let proof_path = format!("{}/{}-{}.proof.bin", cache_folder, table_id, filter_info);
        let _ = fs::create_dir_all(&cache_folder);
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
