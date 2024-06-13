use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};

use afs_chips::{
    execution_air::ExecutionAir,
    page_rw_checker::page_controller::{OpType, Operation, PageController},
};
use afs_stark_backend::{
    keygen::types::MultiStarkPartialProvingKey,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use alloy_primitives::{Uint, U256};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions, AfsOperation},
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::{
        codec::{self, fixed_bytes::FixedBytesCodec},
        types::TableMetadata,
        Table,
    },
    types::{Data, Index},
    utils::string_to_fixed_bytes_be_vec,
};
use p3_util::log2_strict_usize;

use crate::common::config::Config;

/// `afs prove` command
/// Uses information from config.toml to generate a proof of the changes made by a .afi file to a table
/// saves the proof in `output-folder` as */prove.bin.
#[derive(Debug, Parser)]
pub struct ProveCommand {
    #[arg(long = "table-id", short = 't', help = "The table ID", required = true)]
    pub table_id: String,

    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "DB file input (default: new empty DB)",
        required = false
    )]
    pub db_file_path: Option<String>,

    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "output"
    )]
    pub output_folder: String,
}

impl ProveCommand {
    /// Execute the `prove` command
    pub fn execute(self, config: &Config) -> Result<()> {
        println!("Proving ops file: {}", self.afi_file_path);
        // prove::prove_ops(&self.ops_file).await?;
        let instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        let mut db = if let Some(db_file_path) = self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(&db_file_path)
        } else {
            let default_table_metadata = TableMetadata::new(32, 1024);
            MockDb::new(default_table_metadata)
        };
        let idx_len = db.default_table_metadata.index_bytes;
        let data_len = db.default_table_metadata.data_bytes;
        assert!(idx_len == config.page.idx_len as usize);
        assert!(data_len == config.page.data_len as usize);
        let codec = FixedBytesCodec::new(idx_len, db.default_table_metadata.data_bytes);
        let mut interface = AfsInterface::new(&mut db);
        let zk_ops = instructions
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| afi_op_conv(op, self.table_id.clone(), &mut interface, i + 1, &codec))
            .collect::<Vec<_>>();

        let page_height = 10000;
        assert!(page_height > 0);
        let page_bus_index = 0;
        let checker_final_bus_index = 1;
        let range_bus_index = 2;
        let ops_bus_index = 3;

        let page_width = 1 + idx_len + data_len;

        let checker_trace_degree = config.page.max_rw_ops as usize * 4;

        let idx_limb_bits = config.schema.limb_size as usize;

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(page_height))
            .max(8);

        let idx_decomp = 8;

        let page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            page_bus_index,
            checker_final_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        let (page_traces, mut prover_data) = page_controller.load_page_and_ops(
            page_init.clone(),
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
            zk_ops.clone(),
            checker_trace_degree,
            &mut trace_builder.committer,
        );

        let offline_checker_trace = page_controller.offline_checker_trace();
        let final_page_aux_trace = page_controller.final_page_aux_trace();
        let range_checker_trace = page_controller.range_checker_trace();

        // Generating trace for ops_sender and making sure it has height num_ops
        let ops_sender_trace =
            ops_sender.generate_trace_testing(&zk_ops, config.page.max_rw_ops as usize, 1);

        // Clearing the range_checker counts
        page_controller.update_range_checker(idx_decomp);

        trace_builder.clear();

        trace_builder.load_cached_trace(page_traces[0].clone(), prover_data.remove(0));
        trace_builder.load_cached_trace(page_traces[1].clone(), prover_data.remove(0));
        trace_builder.load_trace(final_page_aux_trace);
        trace_builder.load_trace(offline_checker_trace.clone());
        trace_builder.load_trace(range_checker_trace);
        trace_builder.load_trace(ops_sender_trace);

        trace_builder.commit_current();
        let encoded_pk = read_from_path(self.output_folder.clone() + "/partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_pk).unwrap();
        let partial_vk = partial_pk.partial_vk();
        let main_trace_data = trace_builder.view(
            &partial_vk,
            vec![
                &page_controller.init_chip,
                &page_controller.final_chip,
                &page_controller.offline_checker,
                &page_controller.range_checker.air,
                &ops_sender,
            ],
        );

        let pis = vec![vec![]; partial_vk.per_air.len()];

        let prover = engine.prover();

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pis);
        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        Ok(())
    }
}

fn afi_op_conv<I, D>(
    afi_op: &AfsOperation,
    table_id: String,
    interface: &mut AfsInterface<I, D>,
    clk: usize,
    codec: &FixedBytesCodec<I, D>,
) -> Operation
where
    I: Index,
    D: Data,
{
    let idx_u8 = string_to_fixed_bytes_be_vec(afi_op.args[0].clone(), 32);
    let idx_u16 = vec_u8_to_vec_u16(&idx_u8);
    let idx = codec.fixed_bytes_to_index(idx_u8.clone());
    match afi_op.operation {
        InputFileBodyOperation::Read => {
            assert!(afi_op.args.len() == 1);
            let data = interface
                .read(table_id, codec.fixed_bytes_to_index(idx_u8))
                .unwrap();
            let data_bytes = codec.data_to_fixed_bytes(data.clone());
            let data_u16 = vec_u8_to_vec_u16(&data_bytes);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Read,
            }
        }
        InputFileBodyOperation::Insert => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_fixed_bytes_be_vec(afi_op.args[1].clone(), D::MEMORY_SIZE);
            let data_u16 = vec_u8_to_vec_u16(&data_u8);
            let data = codec.fixed_bytes_to_data(data_u8);
            interface.insert(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        InputFileBodyOperation::Write => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_fixed_bytes_be_vec(afi_op.args[1].clone(), D::MEMORY_SIZE);
            let data_u16 = vec_u8_to_vec_u16(&data_u8);
            let data = codec.fixed_bytes_to_data(data_u8);
            interface.insert(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
    }
}

// while the output is u32, each element is less than 2 << 16.
fn vec_u8_to_vec_u16(bytes: &Vec<u8>) -> Vec<u32> {
    let mut ans = vec![];
    for i in 0..(bytes.len() + 1) / 2 {
        if bytes.len() % 2 == 1 && i == 0 {
            ans.push(bytes[0] as u32);
        } else {
            ans.push((256 * bytes[2 * i + 1] + bytes[2 * i]) as u32);
        }
    }
    ans
}

// fn table_to_page

fn read_from_path(path: String) -> Option<Vec<u8>> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buf = vec![];
    reader.read_to_end(&mut buf).unwrap();
    Some(buf)
}

fn write_bytes(bytes: &Vec<u8>, path: String) -> Result<()> {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write(bytes).unwrap();
    Ok(())
}
