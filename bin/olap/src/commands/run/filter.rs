use std::{
    fs::File,
    io::{BufWriter, Write},
};

use afs_chips::single_page_index_scan::page_controller::PageController;
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{config, page_config::PageConfig};
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::operation::FilterOp,
    afs_interface::AfsInterface,
    mock_db::MockDb,
    utils::{fixed_bytes_to_u16_vec, string_to_u8_vec},
};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;

use super::RunCommand;

const PAGE_BUS_INDEX: usize = 0;
const RANGE_BUS_INDEX: usize = 1;

pub fn execute_filter_op<SC: StarkGenericConfig>(
    cfg: &PageConfig,
    cli: &RunCommand,
    db: &mut MockDb,
    op: FilterOp,
) -> Result<()> {
    println!("filter: {:?}", op);

    let height = cfg.page.height;
    let bits_per_fe = cfg.page.bits_per_fe;
    let degree = log2_strict_usize(height);

    // Get input page from database
    let interface = AfsInterface::new_with_table(op.table_id.to_string(), db);
    let input_table = interface.current_table().unwrap();
    let index_bytes = input_table.metadata.index_bytes;
    let data_bytes = input_table.metadata.data_bytes;
    let input_page = input_table.to_page(index_bytes, data_bytes, height);
    let index_len = input_page.idx_len();
    let data_len = input_page.data_len();
    let page_width = 1 + index_len + data_len;

    if !cli.silent {
        println!("Input page:");
        input_page.pretty_print(bits_per_fe);
    }

    let mut page_controller = PageController::new(
        PAGE_BUS_INDEX,
        RANGE_BUS_INDEX,
        index_len,
        data_len,
        height as u32,
        bits_per_fe,
        degree,
        op.predicate.clone(),
    );

    let value = string_to_u8_vec(op.value, index_bytes);
    let value = fixed_bytes_to_u16_vec(value);
    let output_page =
        page_controller.gen_output(input_page.clone(), value.clone(), page_width, op.predicate);

    if !cli.silent {
        println!("Output page:");
        output_page.pretty_print(bits_per_fe);
    }

    let engine = config::baby_bear_poseidon2::default_engine(degree);
    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let (input_prover_data, output_prover_data) = page_controller.load_page(
        input_page.clone(),
        output_page.clone(),
        None,
        None,
        value.clone(),
        index_len,
        data_len,
        bits_per_fe,
        degree,
        &mut trace_builder.committer,
    );

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
    page_controller.set_up_keygen_builder(
        &mut keygen_builder,
        page_width,
        height,
        index_len,
        degree,
    );

    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();
    let proof = page_controller.prove(
        &engine,
        &partial_pk,
        &mut trace_builder,
        input_prover_data,
        output_prover_data,
        value.clone(),
        degree,
    );

    let output_path = if let Some(output_path) = &cli.output_path {
        output_path.to_owned()
    } else {
        "bin/olap/tests/data/where.proof.bin".to_string()
    };
    let encoded_proof = bincode::serialize(&proof).unwrap();
    page_controller
        .verify(&engine, partial_vk, proof, value.clone())
        .unwrap();
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(&encoded_proof).unwrap();

    Ok(())
}
