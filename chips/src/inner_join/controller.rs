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

pub struct InnerJoinController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub t1_chip: MyInitialTableAir,
    pub t2_chip: MyInitialTableAir,
    pub output_chip: MyFinalTableAir,
    pub intersector_chip: IntersectorAir,

    table_traces: Option<IJTraces<Val<SC>>>,
    table_commitments: Option<TableCommitments<SC>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> InnerJoinController<SC> {
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
                Val::<SC>::bits() - 1,
                decomp,
            ),

            table_traces: None,
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

        self.table_traces = Some(IJTraces {
            t1_main_trace,
            t1_aux_trace,
            t2_main_trace,
            t2_aux_trace,
            output_main_trace,
            output_aux_trace,
            intersector_trace,
        });

        (self.table_traces.clone().unwrap(), prover_data)
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
