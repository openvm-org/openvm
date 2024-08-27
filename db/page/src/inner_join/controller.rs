use std::{collections::HashMap, iter, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder, TraceCommitter},
        types::Proof,
    },
    verifier::VerificationError,
};
use afs_test_utils::engine::StarkEngine;
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use super::{
    final_table::FinalTableAir,
    initial_table::{InitialTableAir, TableType},
    intersector::IntersectorAir,
};
use crate::{
    common::page::Page,
    inner_join::{final_table::FinalTableBuses, intersector::IntersectorBuses},
};

/// A struct to keep track of the traces of the chips
/// owned by the inner join controller
#[derive(Clone, Serialize, Deserialize)]
pub struct IJTraces<F: AbstractField> {
    pub t1_main_trace: DenseMatrix<F>,
    pub t1_aux_trace: DenseMatrix<F>,
    pub t2_main_trace: DenseMatrix<F>,
    pub t2_aux_trace: DenseMatrix<F>,
    pub output_main_trace: DenseMatrix<F>,
    pub output_aux_trace: DenseMatrix<F>,
    pub intersector_trace: DenseMatrix<F>,
}

#[allow(dead_code)]
struct TableCommitments<SC: StarkGenericConfig> {
    t1_commitment: Com<SC>,
    t2_commitment: Com<SC>,
    output_commitment: Com<SC>,
}

/// A struct containing all bus indices used by the inner join controller
pub struct IJBuses {
    pub range_bus_index: usize,
    pub t1_intersector_bus_index: usize,
    pub t2_intersector_bus_index: usize,
    pub intersector_t2_bus_index: usize,
    pub t1_output_bus_index: usize,
    pub t2_output_bus_index: usize,
}

/// A struct containing the basic format of the tables
#[derive(Clone, derive_new::new)]
pub struct TableFormat {
    pub idx_len: usize,
    pub data_len: usize,
    pub idx_limb_bits: usize,
}

/// A struct containing the format of the T2 table (Child Table)
#[derive(Clone, derive_new::new)]
pub struct T2Format {
    pub table_format: TableFormat,
    pub fkey_start: usize,
    pub fkey_end: usize,
}

/// This is a controller the Inner Join operation on tables T1 (with primary key) and T2 (which foreign key).
/// This controller owns four chips: t1_chip, t2_chip, output_chip, and intersector_chip. A trace partition
/// of t1_chip is T1, a trace partition of t2_chip is T2, and a trace partition of output_chip is the output table.
/// The goal is to prove that the output table is the result of performing the Inner Join operation on T1 and T2.
///
/// Note that we assume that T1 and T2 are given in a proper format: allocated rows come first, indices in allocated
/// rows are sorted and distinct, and unallocated rows are all zeros.
///
/// High level overview:
/// We do this by introducing the intersector_chip, which helps us verify the multiplicity of each index as foreign key in
/// the output table. The intersector chip receives all primary keys in T1 (with t1_mult) and all foreign keys in T2 (with t2_mult).
/// It then computes out_mult as t1_mult*t2_mult, which should be exactly the number of times index appears (in place of foreign key)
/// in the output table. The intersector chip then sends each index with multiplicity out_mult to the t2_chip, and this allows the t2
/// chip to verify if each row makes it to the output table or not.
/// Using this information, t1_chip and t2_chip then send the necessary data to the output_chip to verify the correctness
/// of the output table.
/// Note that we use different buses for those interactions to ensure soundness.
///
/// Exact protocol:
/// We have four chips: one for T1, one for T2, one for the output table, and one helper we call the intersector chip.
/// The traces for T1 and T2 should be cached. We will use five buses: T1_intersector, T2_intersector, intersector_T2, T1_output, and T2_output
/// (bus a_b is sent to by only a and received from by only b). Here is an outline of the interactions and the constraints:
/// - T1 sends primary key (idx) on T1_intersector with multiplicity is_alloc
/// - T2 sends foreign key on T2_intersector with multiplicity is_alloc
/// - The intersector chip should do the following:
///     - Every row in the trace has an index (of width idx_len of T1) and a few extra columns: T1_mult, T2_mult, and out_mult.
///     - There should be a row for every index that appears as a primary key of T1 or a foreign key in T2
///     - Receives idx with multiplicity T1_mult on T1_intersector bus
///     - Receives idx with multiplicity T2_mult on T2_intersector bus
///     - out_mult should be the multiplication of T1_mult and T2_mult
///     - Sends idx with multiplicity out_mult on intersector_T2 bus
///     - The indices in the trace should be sorted in strict increasing order (using the less than chip).
///         - This is important to make sure the out_mult is calculated correctly for every idx
/// - T2 should have an extra column fkey_present in another partition of the trace. The value in that column
///   should be 1 if the foreign key in the row of T2 appears in T1 as a primary key, and it should be 0 otherwise
/// - T2 receives foreign key with multiplicity fkey_present on intersector_T2 bus
/// - T2 sends each row (idx and data) with multiplicity fkey_present on T2_output bus
/// - T1 should have an extra column out_mult which should be the number of times the primary key in that row appears in the output
/// - T1 sends each row (idx and data) with multiplicity out_mult on T1_output bus
/// - Output page receives idx and data of T1 on T1_output bus with multiplicity is_alloc
/// - Output page receives idx and data of T2 on T2_output bus with multiplicity is_alloc (Note that the this receive
///   shares with the previous receive the same columns that correspond to the key of T1)
/// - We need to ensure that all the multiplicity columns (out_mult in T1, fkey_present in T2, out_mult in intersector chip)
///   are 0 if is_alloc or is_extra (described below) is 0.
pub struct FKInnerJoinController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    t1_chip: InitialTableAir,
    t2_chip: InitialTableAir,
    output_chip: FinalTableAir,
    intersector_chip: IntersectorAir,

    traces: Option<IJTraces<Val<SC>>>,
    table_commitments: Option<TableCommitments<SC>>,

    range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> FKInnerJoinController<SC> {
    /// Note that here we refer to the Parent Table (or the Referenced Table) as T1 and
    /// the Child Table (or the Referencing Table) as T2
    /// [fkey_start, fkey_end) is the range of the foreign key within the data part of T2
    pub fn new(buses: IJBuses, t1_format: TableFormat, t2_format: T2Format, decomp: usize) -> Self
    where
        Val<SC>: Field,
    {
        // Ensuring the foreign key range is valid
        assert!(
            t2_format.fkey_start < t2_format.fkey_end
                && t2_format.fkey_end <= t2_format.table_format.data_len
        );

        Self {
            t1_chip: InitialTableAir::new(
                t1_format.idx_len,
                t1_format.data_len,
                TableType::T1 {
                    t1_intersector_bus_index: buses.t1_intersector_bus_index,
                    t1_output_bus_index: buses.t1_output_bus_index,
                },
            ),
            t2_chip: InitialTableAir::new(
                t2_format.table_format.idx_len,
                t2_format.table_format.data_len,
                TableType::new_t2(
                    t2_format.fkey_start,
                    t2_format.fkey_end,
                    buses.t2_intersector_bus_index,
                    buses.intersector_t2_bus_index,
                    buses.t2_output_bus_index,
                ),
            ),
            output_chip: FinalTableAir::new(
                FinalTableBuses::new(buses.t1_output_bus_index, buses.t2_output_bus_index),
                buses.range_bus_index,
                t1_format.clone(),
                t2_format,
                decomp,
            ),
            intersector_chip: IntersectorAir::new(
                buses.range_bus_index,
                IntersectorBuses::new(
                    buses.t1_intersector_bus_index,
                    buses.t2_intersector_bus_index,
                    buses.intersector_t2_bus_index,
                ),
                t1_format.idx_len,
                Val::<SC>::bits() - 1, // Here, we use the full range of the field because there's no guarantee that the foreign key is in the idx_limb_bits range
                decomp,
            ),

            traces: None,
            table_commitments: None,

            range_checker: Arc::new(RangeCheckerGateChip::new(
                buses.range_bus_index,
                1 << decomp,
            )),
        }
    }

    /// This function creates a new range checker (using decomp).
    /// Helpful for clearing range_checker counts
    pub fn reset_range_checker(&mut self, decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            1 << decomp,
        ));
    }

    /// This function generates the main traces for the input and output tables.
    /// It returns a tuple of three RowMajorMatrix objects, representing the main
    /// traces for T1, T2, and the output table, respectively.
    #[allow(clippy::type_complexity)]
    pub fn io_main_traces(
        &mut self,
        t1: &Page,
        t2: &Page,
    ) -> (
        RowMajorMatrix<Val<SC>>,
        RowMajorMatrix<Val<SC>>,
        RowMajorMatrix<Val<SC>>,
    )
    where
        Val<SC>: PrimeField,
    {
        let (output_table, _fkey_start, _fkey_end) = self.calc_output_table(t1, t2);

        let t1_main_trace = self.gen_table_trace(t1);
        let t2_main_trace = self.gen_table_trace(t2);
        let output_main_trace = self.gen_table_trace(&output_table);

        (t1_main_trace, t2_main_trace, output_main_trace)
    }

    /// This function manages the trace generation of the different chips to necessary
    /// for the inner join operation on T1 and T2. It creates the output_table, which
    /// is the result of the inner join operation, calls the trace generation for the
    /// the actual tables (T1, T2, output_table) and for the auxiliary traces for the
    /// tables (mainly used for the interactions). It also calls the trace generation
    /// for the intersector_chip.
    ///
    /// Returns ProverTraceData for the actual tables (T1, T2, output_tables). Moreover,
    /// a copy of the traces and the commitments for the tables is stored in the struct.
    #[allow(clippy::too_many_arguments)]
    pub fn load_tables(
        &mut self,
        t1: &Page,
        t2: &Page,
        page1_input_pdata: Option<ProverTraceData<SC>>,
        page2_input_pdata: Option<ProverTraceData<SC>>,
        page_output_pdata: Option<ProverTraceData<SC>>,
        intersector_trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> Vec<ProverTraceData<SC>>
    where
        Val<SC>: PrimeField,
    {
        let (output_table, fkey_start, fkey_end) = self.calc_output_table(t1, t2);
        let (t1_main_trace, t2_main_trace, output_main_trace) = self.io_main_traces(t1, t2);

        // Calculating the multiplicity with which T1 and T2 indices appear in the output_table
        let mut t1_idx_out_mult = HashMap::new();
        let mut t2_idx_out_mult = HashMap::new();
        for row in output_table.iter() {
            if row.is_alloc == 1 {
                t1_idx_out_mult
                    .entry(row.data[fkey_start..fkey_end].to_vec())
                    .and_modify(|e| *e += 1)
                    .or_insert(1);

                t2_idx_out_mult
                    .entry(row.idx.clone())
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
            }
        }

        let mut t1_out_mult = vec![];
        for row in t1.iter() {
            if row.is_alloc == 1 {
                t1_out_mult.push(*t1_idx_out_mult.get(&row.idx).unwrap_or(&0));
            } else {
                t1_out_mult.push(0);
            }
        }

        let mut t2_fkey_present = vec![];
        for row in t2.iter() {
            if row.is_alloc == 1 {
                t2_fkey_present.push(*t2_idx_out_mult.get(&row.idx).unwrap_or(&0));
            } else {
                t2_fkey_present.push(0);
            }
        }

        let t1_aux_trace = self.t1_chip.gen_aux_trace(&t1_out_mult);
        let t2_aux_trace = self.t2_chip.gen_aux_trace(&t2_fkey_present);
        let intersector_trace = self.intersector_chip.generate_trace(
            t1,
            t2,
            fkey_start,
            fkey_end,
            self.range_checker.clone(),
            intersector_trace_degree,
        );
        let output_aux_trace = self
            .output_chip
            .gen_aux_trace::<SC>(&output_table, self.range_checker.clone());

        // Commit the traces if they are not provided
        let t1_commit = page1_input_pdata
            .unwrap_or_else(|| trace_committer.commit(vec![t1_main_trace.clone()]));
        let t2_commit = page2_input_pdata
            .unwrap_or_else(|| trace_committer.commit(vec![t2_main_trace.clone()]));
        let output_commit = page_output_pdata
            .unwrap_or_else(|| trace_committer.commit(vec![output_main_trace.clone()]));
        let prover_data = vec![t1_commit, t2_commit, output_commit];

        self.table_commitments = Some(TableCommitments {
            t1_commitment: prover_data[0].commit.clone(),
            t2_commitment: prover_data[1].commit.clone(),
            output_commitment: prover_data[2].commit.clone(),
        });

        self.traces = Some(IJTraces {
            t1_main_trace,
            t1_aux_trace,
            t2_main_trace,
            t2_aux_trace,
            output_main_trace,
            output_aux_trace,
            intersector_trace,
        });

        prover_data
    }

    /// Sets up keygen with the different trace partitions for all the
    /// chips the struct owns (t1_chip, t2_chip, output_chip, intersector_chip)
    pub fn set_up_keygen_builder<'a>(&'a self, keygen_builder: &mut MultiStarkKeygenBuilder<'a, SC>)
    where
        Val<SC>: PrimeField,
    {
        let t1_main_ptr = keygen_builder.add_cached_main_matrix(self.t1_chip.table_width());
        let t2_main_ptr = keygen_builder.add_cached_main_matrix(self.t2_chip.table_width());
        let output_main_ptr = keygen_builder.add_cached_main_matrix(self.output_chip.table_width());
        let t1_aux_ptr = keygen_builder.add_main_matrix(self.t1_chip.aux_width());
        let t2_aux_ptr = keygen_builder.add_main_matrix(self.t2_chip.aux_width());
        let output_aux_ptr = keygen_builder.add_main_matrix(self.output_chip.aux_width());

        keygen_builder.add_partitioned_air(&self.t1_chip, 0, vec![t1_main_ptr, t1_aux_ptr]);

        keygen_builder.add_partitioned_air(&self.t2_chip, 0, vec![t2_main_ptr, t2_aux_ptr]);

        keygen_builder.add_partitioned_air(
            &self.output_chip,
            0,
            vec![output_main_ptr, output_aux_ptr],
        );

        keygen_builder.add_air(&self.intersector_chip, 0);
        keygen_builder.add_air(&self.range_checker.air, 0);
    }

    /// This function clears the trace_builder, loads in the traces for all involved chips
    /// (including the range_checker), commits them, and then generates the proof.
    /// cached_traces_prover_data is a vector of ProverTraceData object for the cached tables
    /// (T1, T2, output_table in that order)
    pub fn prove(
        &self,
        engine: &impl StarkEngine<SC>,
        pk: &MultiStarkProvingKey<SC>,
        trace_builder: &mut TraceCommitmentBuilder<SC>,
        mut cached_traces_prover_data: Vec<ProverTraceData<SC>>,
    ) -> Proof<SC>
    where
        Val<SC>: PrimeField,
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        assert!(cached_traces_prover_data.len() == 3);

        let traces = self.traces.as_ref().unwrap();

        trace_builder.clear();

        trace_builder.load_cached_trace(
            traces.t1_main_trace.clone(),
            cached_traces_prover_data.remove(0),
        );
        trace_builder.load_cached_trace(
            traces.t2_main_trace.clone(),
            cached_traces_prover_data.remove(0),
        );
        trace_builder.load_cached_trace(
            traces.output_main_trace.clone(),
            cached_traces_prover_data.remove(0),
        );
        trace_builder.load_trace(traces.t1_aux_trace.clone());
        trace_builder.load_trace(traces.t2_aux_trace.clone());
        trace_builder.load_trace(traces.output_aux_trace.clone());
        trace_builder.load_trace(traces.intersector_trace.clone());
        trace_builder.load_trace(self.range_checker.generate_trace());

        trace_builder.commit_current();

        let vk = pk.vk();

        let main_trace_data = trace_builder.view(
            &vk,
            vec![
                &self.t1_chip,
                &self.t2_chip,
                &self.output_chip,
                &self.intersector_chip,
                &self.range_checker.air,
            ],
        );

        let pis = vec![vec![]; vk.per_air.len()];

        let prover = engine.prover();

        let mut challenger = engine.new_challenger();
        prover.prove(&mut challenger, pk, main_trace_data, &pis)
    }

    /// This function takes a proof (returned by the prove function) and verifies it
    pub fn verify(
        &self,
        engine: &impl StarkEngine<SC>,
        vk: MultiStarkVerifyingKey<SC>,
        proof: Proof<SC>,
    ) -> Result<(), VerificationError>
    where
        Val<SC>: PrimeField,
    {
        let verifier = engine.verifier();

        let pis = vec![vec![]; vk.per_air.len()];

        let mut challenger = engine.new_challenger();
        verifier.verify(&mut challenger, &vk, &proof, &pis)
    }

    pub fn traces(&self) -> Option<&IJTraces<Val<SC>>> {
        self.traces.as_ref()
    }

    /// This function takes two tables T1 and T2 and the range of the foreign key in T2
    /// It returns the Page resulting from the inner join operations on those parameters
    fn inner_join(&self, t1: &Page, t2: &Page, fkey_start: usize, fkey_end: usize) -> Page {
        let mut output_table = vec![];

        for row in t2.iter() {
            if row.is_alloc == 0 {
                continue;
            }

            let fkey = row.data[fkey_start..fkey_end].to_vec();
            if !t1.contains(&fkey) {
                continue;
            } else {
                let out_row: Vec<u32> = iter::once(1)
                    .chain(row.idx.clone())
                    .chain(row.data.clone())
                    .chain(t1[&fkey].clone())
                    .collect();

                output_table.push(out_row);
            }
        }

        // Padding the output page with unallocated rows so that it has the same height as t2
        output_table.resize(
            t2.height(),
            vec![0; 1 + t2.idx_len() + t2.data_len() + t1.data_len()],
        );

        Page::from_2d_vec(&output_table, t2.idx_len(), t2.data_len() + t1.data_len())
    }

    fn gen_table_trace(&self, page: &Page) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        page.gen_trace::<Val<SC>>()
    }

    fn calc_output_table(&self, t1: &Page, t2: &Page) -> (Page, usize, usize) {
        let (fkey_start, fkey_end) = match self.t2_chip.table_type {
            TableType::T2 {
                fkey_start,
                fkey_end,
                ..
            } => (fkey_start, fkey_end),
            _ => panic!("t2 must be of TableType T2"),
        };

        let output_table = self.inner_join(t1, t2, fkey_start, fkey_end);

        (output_table, fkey_start, fkey_end)
    }
}
