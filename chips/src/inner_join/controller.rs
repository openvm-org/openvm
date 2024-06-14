use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{
    initial_table::{MyInitialTableAir, TableType},
    intersector::IntersectorAir,
};
use crate::common::page::Page;
use crate::final_page::FinalPageAir;
use crate::range_gate::RangeCheckerGateChip;

#[derive(Clone)]
#[allow(dead_code)]
pub struct TableTraces<F: AbstractField> {
    t1_trace: DenseMatrix<F>,
    t2_trace: DenseMatrix<F>,
    output_main_trace: DenseMatrix<F>,
    output_aux_trace: DenseMatrix<F>,
    intersector_trace: DenseMatrix<F>,
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
    pub final_chip: FinalPageAir,
    pub intersector_chip: IntersectorAir,

    table_traces: Option<TableTraces<Val<SC>>>,
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
        idx_decomp: usize,
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
            final_chip: FinalPageAir::new(
                range_bus_index,
                t2_idx_len,
                t1_data_len + t2_data_len,
                idx_limb_bits,
                idx_decomp,
            ),
            intersector_chip: IntersectorAir::new(
                t1_intersector_bus_index,
                t2_intersector_bus_index,
                intersector_t2_bus_index,
                t1_idx_len,
            ),

            table_traces: None,
            table_commitments: None,

            range_checker: Arc::new(RangeCheckerGateChip::new(range_bus_index, 1 << idx_decomp)),
        }
    }

    fn gen_table_trace(&self, page: &Page) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        page.gen_trace::<Val<SC>>()
    }

    pub fn update_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            1 << idx_decomp,
        ));
    }

    pub fn load_tables(
        &mut self,
        t1: &Page,
        t2: &Page,
        intersector_trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (TableTraces<Val<SC>>, Vec<ProverTraceData<SC>>)
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

        let t1_trace = self.gen_table_trace(t1);
        let t2_trace = self.gen_table_trace(t2);
        let intersector_trace = self.intersector_chip.generate_trace(
            t1,
            t2,
            fkey_start,
            fkey_end,
            intersector_trace_degree,
        );
        let output_main_trace = self.gen_table_trace(&output_table);
        let output_aux_trace = self
            .final_chip
            .gen_aux_trace::<SC>(&output_table, self.range_checker.clone());

        let prover_data = vec![
            trace_committer.commit(vec![t1_trace.clone()]),
            trace_committer.commit(vec![t2_trace.clone()]),
            trace_committer.commit(vec![output_main_trace.clone()]),
        ];

        self.table_commitments = Some(TableCommitments {
            t1_commitment: prover_data[0].commit.clone(),
            t2_commitment: prover_data[1].commit.clone(),
            output_commitment: prover_data[2].commit.clone(),
        });

        self.table_traces = Some(TableTraces {
            t1_trace,
            t2_trace,
            output_main_trace,
            output_aux_trace,
            intersector_trace,
        });

        (self.table_traces.clone().unwrap(), prover_data)
    }

    fn inner_join(&self, t1: &Page, t2: &Page, fkey_start: usize, fkey_end: usize) -> Page {
        let mut output_table = vec![];

        let mut t1_map: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
        for row in t1.rows.iter() {
            t1_map.insert(row.idx.clone(), row.data.clone());
        }

        for row in t2.rows.iter() {
            if row.is_alloc == 0 {
                continue;
            }

            let fkey = row.data[fkey_start..fkey_end].to_vec();
            if t1_map.contains_key(&fkey) == false {
                continue;
            } else {
                let out_row: Vec<u32> = iter::once(1)
                    .chain(row.idx.clone())
                    .chain(row.data.clone())
                    .chain(t1_map[&fkey].clone())
                    .collect();

                output_table.push(out_row);
            }
        }

        Page::from_2d_vec(&output_table, t2.idx_len(), t2.data_len() + t1.data_len())
    }
}
