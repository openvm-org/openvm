use std::{iter, sync::Arc};

use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{
    intersector::IntersectorAir,
    my_final_table::MyFinalTableAir,
    my_initial_table::{MyInitialTableAir, TableType},
};
use crate::{common::page::Page, range_gate::RangeCheckerGateChip};

/// A struct to keep track of the traces of the chips
/// owned by the inner join controller
#[derive(Clone)]
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
pub struct InnerJoinController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub t1_chip: MyInitialTableAir,
    pub t2_chip: MyInitialTableAir,
    pub output_chip: MyFinalTableAir,
    pub intersector_chip: IntersectorAir,

    traces: Option<IJTraces<Val<SC>>>,
    table_commitments: Option<TableCommitments<SC>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> InnerJoinController<SC> {
    /// [fkey_start, fkey_end) is the range of the foreign key within the data part of T2
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        range_bus_index: usize,
        t1_intersector_bus_index: usize,
        t2_intersector_bus_index: usize,
        intersector_t2_bus_index: usize,
        t1_output_bus_index: usize,
        t2_output_bus_index: usize,
        fkey_start: usize,
        fkey_end: usize,
        t1_idx_len: usize,
        t1_data_len: usize,
        t2_idx_len: usize,
        t2_data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
    ) -> Self
    where
        Val<SC>: Field,
    {
        Self {
            t1_chip: MyInitialTableAir::new(
                t1_idx_len,
                t1_data_len,
                TableType::T1 {
                    t1_intersector_bus_index,
                    t1_output_bus_index,
                },
            ),
            t2_chip: MyInitialTableAir::new(
                t2_idx_len,
                t2_data_len,
                TableType::T2 {
                    fkey_start,
                    fkey_end,
                    t2_intersector_bus_index,
                    intersector_t2_bus_index,
                    t2_output_bus_index,
                },
            ),
            output_chip: MyFinalTableAir::new(
                t1_output_bus_index,
                t2_output_bus_index,
                range_bus_index,
                t2_idx_len,
                t1_data_len,
                t2_data_len,
                fkey_start,
                fkey_end,
                idx_limb_bits,
                decomp,
            ),
            intersector_chip: IntersectorAir::new(
                range_bus_index,
                t1_intersector_bus_index,
                t2_intersector_bus_index,
                intersector_t2_bus_index,
                t1_idx_len,
                Val::<SC>::bits() - 1, // Here, we use the full range of the field because there's no guarantee that the foreign key is in the idx_limb_bits range
                decomp,
            ),

            traces: None,
            table_commitments: None,

            range_checker: Arc::new(RangeCheckerGateChip::new(range_bus_index, 1 << decomp)),
        }
    }

    /// This function creates a new range checker (using decomp).
    /// Helpful for clearing range_checker counts
    pub fn update_range_checker(&mut self, decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            1 << decomp,
        ));
    }

    /// This function manages the trace generation of the different chips to necessary
    /// for the inner join operation on T1 and T2. It creates the output_table, which
    /// is the result of the inner join operation, calls the trace generation for the
    /// the actual tables (T1, T2, output_table) and for the auxiliary traces for the
    /// tables (mainly used for the interactions). It also calls the trace generation
    /// for the intersector_chip.
    ///
    /// Returns the traces T1, T2, output_table, and the intersector_chip, along with
    /// the ProverTraceData for the actual tables (T1, T2, output_tables). Moreover,
    /// a copy of the traces and the commitments for the tables is stored in the struct.
    pub fn load_tables(
        &mut self,
        t1: &Page,
        t2: &Page,
        intersector_trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (IJTraces<Val<SC>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        let (fkey_start, fkey_end) = match self.t2_chip.table_type {
            TableType::T2 {
                fkey_start,
                fkey_end,
                ..
            } => (fkey_start, fkey_end),
            _ => panic!("t2 must be of TableType T2"),
        };

        let output_table = self.inner_join(t1, t2, fkey_start, fkey_end);

        // Calculating the multiplicity with which T1 indices appear in the output_table
        let mut t1_out_mult = vec![];
        for row in t1.rows.iter() {
            if row.is_alloc == 1 {
                t1_out_mult.push(
                    output_table
                        .rows
                        .iter()
                        .filter(|out_row| {
                            out_row.is_alloc == 1 && out_row.data[fkey_start..fkey_end] == row.idx
                        })
                        .count() as u32,
                );
            } else {
                t1_out_mult.push(0);
            }
        }

        // Figuring out whether each row of T2 appears in the output_table
        let mut t2_fkey_present = vec![];
        for row in t2.rows.iter() {
            if row.is_alloc == 1 {
                t2_fkey_present.push(
                    output_table
                        .rows
                        .iter()
                        .filter(|out_row| out_row.is_alloc == 1 && out_row.idx == row.idx)
                        .count() as u32,
                );
            } else {
                t2_fkey_present.push(0);
            }
        }

        let t1_main_trace = self.gen_table_trace(t1);
        let t1_aux_trace = self.t1_chip.gen_aux_trace(&t1_out_mult);
        let t2_main_trace = self.gen_table_trace(t2);
        let t2_aux_trace = self.t2_chip.gen_aux_trace(&t2_fkey_present);
        let intersector_trace = self.intersector_chip.generate_trace(
            t1,
            t2,
            fkey_start,
            fkey_end,
            self.range_checker.clone(),
            intersector_trace_degree,
        );
        let output_main_trace = self.gen_table_trace(&output_table);
        let output_aux_trace = self
            .output_chip
            .gen_aux_trace::<SC>(&output_table, self.range_checker.clone());

        let prover_data = vec![
            trace_committer.commit(vec![t1_main_trace.clone()]),
            trace_committer.commit(vec![t2_main_trace.clone()]),
            trace_committer.commit(vec![output_main_trace.clone()]),
        ];

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

        (self.traces.clone().unwrap(), prover_data)
    }

    /// This function takes two tables T1 and T2 and the range of the foreign key in T2
    /// It returns the Page resulting from the inner join operations on those parameters
    fn inner_join(&self, t1: &Page, t2: &Page, fkey_start: usize, fkey_end: usize) -> Page {
        let mut output_table = vec![];

        for row in t2.rows.iter() {
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
}
