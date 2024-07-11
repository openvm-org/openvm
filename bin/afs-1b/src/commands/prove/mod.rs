use std::{sync::Arc, time::Instant};

use afs_chips::{
    execution_air::ExecutionAir,
    multitier_page_rw_checker::page_controller::{
        gen_some_products_from_prover_data, MyLessThanTupleParams, PageController, PageTreeParams,
    },
    page_btree::PageBTree,
    page_rw_checker::page_controller::{OpType, Operation},
    range_gate::RangeCheckerGateChip,
};
use afs_stark_backend::{
    keygen::types::MultiStarkPartialProvingKey,
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder},
        MultiTraceStarkProver,
    },
    rap::AnyRap,
};

use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
    page_config::{MultitierPageConfig, PageMode},
};
use clap::Parser;
use color_eyre::eyre::Result;
use itertools::Itertools;
use logical_interface::{
    afs_input::{
        types::{AfsOperation, InputFileOp},
        AfsInputFile,
    },
    utils::string_to_u16_vec,
};
use p3_baby_bear::BabyBear;
use p3_util::log2_strict_usize;

use crate::commands::{
    commit_to_string, get_prover_data_from_file, read_from_path, write_bytes,
    BABYBEAR_COMMITMENT_LEN, DECOMP_BITS, LIMB_BITS,
};

use super::create_prefix;

/// `afs prove` command
/// Uses information from config.toml to generate a proof of the changes made by a .afi file to a table
/// saves the proof in `output-folder` as */prove.bin.
#[derive(Debug, Parser)]
pub struct ProveCommand {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "db-folder",
        short = 'd',
        help = "Mock DB folder",
        required = false,
        default_value = "multitier_mockdb"
    )]
    pub db_folder: String,

    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains keys",
        required = false,
        default_value = "keys"
    )]
    pub keys_folder: String,

    #[arg(
        long = "cache-folder",
        short = 'c',
        help = "The folder that contains cached traces",
        required = false,
        default_value = "cache"
    )]
    pub cache_folder: String,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

impl ProveCommand {
    /// Execute the `prove` command
    pub fn execute(&self, config: &MultitierPageConfig) -> Result<()> {
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => self.execute_rw(config, prefix)?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Proved table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_rw(&self, config: &MultitierPageConfig, prefix: String) -> Result<()> {
        println!("Proving ops file: {}", self.afi_file_path);

        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputFile::open(&self.afi_file_path)?;
        let table_id = instructions.header.table_id.clone();
        let dst_id = table_id.clone() + ".0";
        let mut db = PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(
            self.db_folder.clone(),
            table_id.to_owned(),
            dst_id,
        )
        .unwrap();

        let zk_ops = instructions
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| {
                afi_op_conv(
                    &mut db,
                    op,
                    config.page.index_bytes,
                    config.page.data_bytes,
                    i + 1,
                )
            })
            .collect::<Vec<_>>();
        let db_folder = self.db_folder.clone();
        let idx_len = (config.page.index_bytes + 1) / 2 as usize;
        let data_len = (config.page.data_bytes + 1) / 2 as usize;
        let data_bus_index = 0;
        let internal_data_bus_index = 1;
        let lt_bus_index = 2;

        let page_height = config.page.height;

        let trace_degree = config.page.max_rw_ops * 4;

        let log_page_height = log2_strict_usize(page_height);
        let log_trace_degree = log2_strict_usize(trace_degree);

        let init_path_bus = 3;
        let final_path_bus = 4;
        let ops_bus_index = 5;

        let less_than_tuple_param = MyLessThanTupleParams {
            limb_bits: LIMB_BITS,
            decomp: DECOMP_BITS,
        };

        let range_checker = Arc::new(RangeCheckerGateChip::new(lt_bus_index, 1 << DECOMP_BITS));

        let mut page_controller: PageController<BABYBEAR_COMMITMENT_LEN> =
            PageController::new::<BabyBearPoseidon2Config>(
                data_bus_index,
                internal_data_bus_index,
                ops_bus_index,
                lt_bus_index,
                idx_len,
                data_len,
                PageTreeParams {
                    path_bus_index: init_path_bus,
                    leaf_cap: config.tree.init_leaf_cap,
                    internal_cap: config.tree.init_internal_cap,
                    leaf_page_height: page_height,
                    internal_page_height: page_height,
                },
                PageTreeParams {
                    path_bus_index: final_path_bus,
                    leaf_cap: config.tree.final_leaf_cap,
                    internal_cap: config.tree.final_internal_cap,
                    leaf_page_height: page_height,
                    internal_page_height: page_height,
                },
                less_than_tuple_param,
                range_checker,
            );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let engine = config::baby_bear_poseidon2::default_engine(
            log_page_height.max(DECOMP_BITS).max(log_trace_degree),
        );
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        db.commit(&mut trace_builder.committer, db_folder.clone());

        let init_pages = db.gen_loaded_trace();
        let final_pages = db.gen_all_trace(&mut trace_builder.committer, None);
        let init_root_is_leaf = init_pages.internal_pages.is_empty();
        let final_root_is_leaf = final_pages.internal_pages.is_empty();

        let encoded_blank_prover_data =
            read_from_path(self.keys_folder.clone() + "/" + &prefix + ".blank_leaf.cache.bin")
                .unwrap();
        let blank_leaf_prover_data: ProverTraceData<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_blank_prover_data).unwrap();

        let encoded_blank_prover_data =
            read_from_path(self.keys_folder.clone() + "/" + &prefix + ".blank_internal.cache.bin")
                .unwrap();
        let blank_internal_prover_data: ProverTraceData<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_blank_prover_data).unwrap();

        let mut init_leaf_prover_data = init_pages
            .leaf_commits
            .iter()
            .map(|l| {
                let s = commit_to_string(l);
                get_prover_data_from_file(db_folder.clone() + "/leaf/" + &s + ".cache.bin")
            })
            .collect_vec();
        let mut init_internal_prover_data = init_pages
            .internal_commits
            .iter()
            .map(|l| {
                let s = commit_to_string(l);
                get_prover_data_from_file(db_folder.clone() + "/internal/" + &s + ".cache.bin")
            })
            .collect_vec();
        let mut final_leaf_prover_data = final_pages
            .leaf_commits
            .iter()
            .map(|l| {
                let s = commit_to_string(l);
                get_prover_data_from_file(db_folder.clone() + "/leaf/" + &s + ".cache.bin")
            })
            .collect_vec();
        let mut final_internal_prover_data = final_pages
            .internal_commits
            .iter()
            .map(|l| {
                let s = commit_to_string(l);
                get_prover_data_from_file(db_folder.clone() + "/internal/" + &s + ".cache.bin")
            })
            .collect_vec();
        init_leaf_prover_data.resize(config.tree.init_leaf_cap, blank_leaf_prover_data.clone());
        init_internal_prover_data.resize(
            config.tree.init_internal_cap,
            blank_internal_prover_data.clone(),
        );
        final_leaf_prover_data.resize(config.tree.final_leaf_cap, blank_leaf_prover_data.clone());
        final_internal_prover_data.resize(
            config.tree.final_internal_cap,
            blank_internal_prover_data.clone(),
        );
        let (data_trace, main_trace, commits, mut prover_data) = page_controller.load_page_and_ops(
            init_pages.leaf_pages,
            init_pages.internal_pages,
            init_root_is_leaf,
            0,
            final_pages.leaf_pages,
            final_pages.internal_pages,
            final_root_is_leaf,
            0,
            &zk_ops,
            trace_degree,
            &mut trace_builder.committer,
            Some((
                gen_some_products_from_prover_data(init_leaf_prover_data),
                gen_some_products_from_prover_data(init_internal_prover_data),
            )),
            Some((
                gen_some_products_from_prover_data(final_leaf_prover_data),
                gen_some_products_from_prover_data(final_internal_prover_data),
            )),
        );
        let offline_checker_trace = main_trace.offline_checker_trace.clone();
        let init_root = main_trace.init_root_signal_trace.clone();
        let final_root = main_trace.final_root_signal_trace.clone();
        let range_trace = page_controller.range_checker.generate_trace();
        let ops_sender_trace = ops_sender.generate_trace(&zk_ops, config.page.max_rw_ops);
        trace_builder.clear();

        for trace in data_trace.init_leaf_chip_traces.iter() {
            trace_builder.load_cached_trace(trace.clone(), prover_data.init_leaf_page.remove(0));
        }

        for trace in data_trace.init_internal_chip_traces.iter() {
            trace_builder
                .load_cached_trace(trace.clone(), prover_data.init_internal_page.remove(0));
        }

        for trace in data_trace.final_leaf_chip_traces.iter() {
            trace_builder.load_cached_trace(trace.clone(), prover_data.final_leaf_page.remove(0));
        }

        for trace in data_trace.final_internal_chip_traces.iter() {
            trace_builder
                .load_cached_trace(trace.clone(), prover_data.final_internal_page.remove(0));
        }

        for trace in main_trace.init_leaf_chip_main_traces.iter() {
            trace_builder.load_trace(trace.clone());
        }

        for trace in main_trace.init_internal_chip_main_traces.iter() {
            trace_builder.load_trace(trace.clone());
        }

        for trace in main_trace.final_leaf_chip_main_traces.iter() {
            trace_builder.load_trace(trace.clone());
        }

        for trace in main_trace.final_internal_chip_main_traces.iter() {
            trace_builder.load_trace(trace.clone());
        }

        trace_builder.load_trace(offline_checker_trace.clone());
        trace_builder.load_trace(init_root);
        trace_builder.load_trace(final_root);
        trace_builder.load_trace(range_trace);
        trace_builder.load_trace(ops_sender_trace);
        trace_builder.commit_current();

        let mut airs: Vec<&dyn AnyRap<BabyBearPoseidon2Config>> = vec![];
        for chip in &page_controller.init_leaf_chips {
            airs.push(chip);
        }
        for chip in &page_controller.init_internal_chips {
            airs.push(chip);
        }
        for chip in &page_controller.final_leaf_chips {
            airs.push(chip);
        }
        for chip in &page_controller.final_internal_chips {
            airs.push(chip);
        }
        airs.push(&page_controller.offline_checker);
        airs.push(&page_controller.init_root_signal);
        airs.push(&page_controller.final_root_signal);
        airs.push(&page_controller.range_checker.air);
        airs.push(&ops_sender);
        let encoded_pk =
            read_from_path(self.keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_pk).unwrap();
        let partial_vk = partial_pk.partial_vk();
        let main_trace_data = trace_builder.view(&partial_vk, airs.clone());

        let mut pis = vec![];
        for c in commits.init_leaf_page_commitments {
            let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        for c in commits.init_internal_page_commitments {
            let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        for c in commits.final_leaf_page_commitments {
            let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        for c in commits.final_internal_page_commitments {
            let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = c.into();
            pis.push(c.to_vec());
        }
        pis.push(vec![]);
        {
            let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = commits.init_root_commitment.into();
            pis.push(c.to_vec());
        }
        {
            let c: [BabyBear; BABYBEAR_COMMITMENT_LEN] = commits.final_root_commitment.into();
            pis.push(c.to_vec());
        }
        pis.push(vec![]);
        pis.push(vec![]);
        let prover = engine.prover();

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pis);
        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let proof_path = self.db_folder.clone() + "/" + &table_id + ".prove.bin";
        let encoded_pis: Vec<u8> = bincode::serialize(&pis).unwrap();
        let pis_path = self.db_folder.clone() + "/" + &table_id + ".pi.bin";
        write_bytes(&encoded_proof, proof_path).unwrap();
        write_bytes(&encoded_pis, pis_path).unwrap();
        Ok(())
    }
}

fn afi_op_conv(
    db: &mut PageBTree<BABYBEAR_COMMITMENT_LEN>,
    afi_op: &AfsOperation,
    idx_bytes: usize,
    data_bytes: usize,
    clk: usize,
) -> Operation {
    let idx_len = (idx_bytes + 1) / 2;
    let data_len = (data_bytes + 1) / 2;
    let idx_u16 = string_to_u16_vec(afi_op.args[0].clone(), idx_len);
    match afi_op.operation {
        InputFileOp::Read => {
            assert!(afi_op.args.len() == 1);
            let data = db.search(&idx_u16).unwrap();
            Operation {
                clk,
                idx: idx_u16,
                data,
                op_type: OpType::Read,
            }
        }
        InputFileOp::Insert => {
            assert!(afi_op.args.len() == 2);
            let data_u16 = string_to_u16_vec(afi_op.args[1].clone(), data_len);
            db.update(&idx_u16, &data_u16);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        InputFileOp::Write => {
            assert!(afi_op.args.len() == 2);
            let data_u16 = string_to_u16_vec(afi_op.args[1].clone(), data_len);
            db.update(&idx_u16, &data_u16);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        _ => panic!(),
    }
}
