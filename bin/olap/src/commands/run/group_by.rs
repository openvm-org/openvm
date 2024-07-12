use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::commands::run::{PageConfig, RunCommand};
use afs_chips::{common::page::Page, group_by::page_controller::PageController};
use afs_stark_backend::{keygen::MultiStarkKeygenBuilder, prover::trace::TraceCommitmentBuilder};
use afs_test_utils::{
    config::baby_bear_poseidon2::{self, BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::operation::GroupByOp, afs_interface::AfsInterface, mock_db::MockDb,
};
use p3_uni_stark::StarkGenericConfig;
use p3_util::log2_strict_usize;

const INTERNAL_BUS: usize = 0;
const OUTPUT_BUS: usize = 1;
const RANGE_BUS: usize = 2;

pub fn execute_group_by<SC: StarkGenericConfig>(
    cfg: &PageConfig,
    cli: &RunCommand,
    db: &mut MockDb,
    op: GroupByOp,
) -> Result<()> {
    if !cli.silent {
        println!("group_by cols: {:?}", op.group_by_cols);
        println!("agg col: {:?}", op.agg_col);
        println!("group operation: {:?}", op.op);
    }

    let index_bytes = cfg.page.index_bytes;
    let data_bytes = cfg.page.data_bytes;
    let height = 16; //cfg.page.height;
    let bits_per_fe = 16; //cfg.page.bits_per_fe;
    let idx_len = (index_bytes + 1) / 2;
    let data_len = (data_bytes + 1) / 2;
    let page_width = 1 + idx_len + data_len;
    let degree = log2_strict_usize(height);
    let decomp = 4;

    // Get page from DB
    let interface = AfsInterface::new_with_table(op.table_id.to_string(), db);
    let table = interface.current_table().unwrap();
    let page = table.to_page(
        table.metadata.index_bytes,
        table.metadata.data_bytes,
        height,
    );

    if !cli.silent {
        println!("Input page");
        page.pretty_print(bits_per_fe);
    }

    let mut page_controller = PageController::<BabyBearPoseidon2Config>::new(
        page_width,
        op.group_by_cols.clone(),
        op.agg_col,
        INTERNAL_BUS,
        OUTPUT_BUS,
        RANGE_BUS,
        bits_per_fe,
        decomp,
        false,
        op.op,
    );
    page_controller.refresh_range_checker();
    let engine = baby_bear_poseidon2::default_engine(degree);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);
    page_controller.set_up_keygen_builder(&mut keygen_builder, height, 1 << decomp);

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    let (group_by_traces, _group_by_commitments, input_prover_data, output_prover_data) =
        page_controller.load_page(&page, None, None, &trace_builder.committer);
    let final_page_trace = group_by_traces.final_page_trace.clone();

    let output_page = Page::from_trace(&final_page_trace, op.group_by_cols.len(), 1);
    if !cli.silent {
        println!("Output page");
        output_page.pretty_print(bits_per_fe);
    }

    let partial_pk = keygen_builder.generate_partial_pk();
    let partial_vk = partial_pk.partial_vk();
    let proof = page_controller.prove(
        &engine,
        &partial_pk,
        &mut trace_builder,
        group_by_traces,
        input_prover_data,
        output_prover_data,
    );

    let output_path = if let Some(output_path) = &cli.output_path {
        output_path.to_owned()
    } else {
        "bin/olap/tests/data/groupby.proof.bin".to_string()
    };
    let encoded_proof = bincode::serialize(&proof).unwrap();
    page_controller.verify(&engine, partial_vk, proof).unwrap();
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(&encoded_proof).unwrap();

    Ok(())
}
