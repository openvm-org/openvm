use std::sync::Arc;

use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::{ProverTraceData, TraceCommitter};
use p3_field::{AbstractField, Field, PrimeField};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{
    intersector::IntersectorAir,
    table::{TableAir, TableType},
};
use crate::final_page::FinalPageAir;
use crate::range_gate::RangeCheckerGateChip;

struct TableCommitments<SC: StarkGenericConfig> {
    t1_trace: DenseMatrix<Val<SC>>,
    t2_trace: DenseMatrix<Val<SC>>,
    output_main_trace: DenseMatrix<Val<SC>>,
    output_aux_trace: DenseMatrix<Val<SC>>,

    t1_commitment: Com<SC>,
    t2_commitment: Com<SC>,
    output_commitment: Com<SC>,
}

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub t1_chip: TableAir,
    pub t2_chip: TableAir,
    pub final_chip: FinalPageAir,

    table_commitments: TableCommitments<SC>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> PageController<SC> {
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
            t1_chip: TableAir::new(
                t1_idx_len,
                t1_data_len,
                TableType::T1 {
                    t1_intersector_bus_index,
                    t1_output_bus_index,
                },
            ),
            t2_chip: TableAir::new(
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

            t1_trace: None,
            t2_trace: None,
            output_main_trace: None,
            output_aux_trace: None,

            t1_commitment: None,
            t2_commitment: None,
            output_commitment: None,

            range_checker: Arc::new(RangeCheckerGateChip::new(range_bus_index, 1 << idx_decomp)),
        }
    }

    pub fn output_aux_trace(&self) -> DenseMatrix<Val<SC>> {
        self.output_aux_trace.clone().unwrap()
    }

    pub fn range_checker_trace(&self) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.range_checker.generate_trace()
    }

    fn gen_table_trace(&self, page: Vec<Vec<u32>>) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.t1_chip.gen_table_trace::<Val<SC>>(page)
    }

    fn gen_ops_trace(
        &self,
        page: &mut Vec<Vec<u32>>,
        ops: &[Operation],
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.offline_checker
            .generate_trace::<SC>(page, ops.to_owned(), range_checker, trace_degree)
    }

    pub fn update_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.air.bus_index,
            1 << idx_decomp,
        ));
    }

    pub fn load_tables(
        &mut self,
        mut t1: Vec<Vec<u32>>,
        mut t2: Vec<Vec<u32>>,
        trace_degree: usize,
        trace_committer: &mut TraceCommitter<SC>,
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        assert!(!t1.is_empty());
        self.t1_trace = Some(self.gen_table_trace(t1.clone()));
        self.t2_trace = Some(self.gen_table_trace(t2.clone()));

        let page_bus_index = self.offline_checker.page_bus_index;
        let range_bus_index = self.offline_checker.range_bus_index;
        let ops_bus_index = self.offline_checker.ops_bus_index;

        self.init_chip = PageAir::new(page_bus_index, idx_len, data_len);
        self.init_chip_trace = Some(self.get_page_trace(page.clone()));

        self.offline_checker = OfflineChecker::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            Val::<SC>::bits() - 1,
            idx_decomp,
        );
        self.offline_checker_trace =
            Some(self.gen_ops_trace(&mut page, &ops, self.range_checker.clone(), trace_degree));

        // Sorting the page by (1-is_alloc, idx)
        page.sort_by_key(|row| (1 - row[0], row[1..1 + idx_len].to_vec()));

        // HashSet of all indices used in operations
        let internal_indices = ops.iter().map(|op| op.idx.clone()).collect();

        self.final_chip = FinalPageAir::new(
            page_bus_index,
            range_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        self.final_chip_trace = Some(self.get_page_trace(page.clone()));
        self.final_page_aux_trace = Some(self.final_chip.gen_aux_trace::<SC>(
            page.clone(),
            self.range_checker.clone(),
            internal_indices,
        ));

        let prover_data = vec![
            trace_committer.commit(vec![self.init_chip_trace.clone().unwrap()]),
            trace_committer.commit(vec![self.final_chip_trace.clone().unwrap()]),
        ];

        self.init_page_commitment = Some(prover_data[0].commit.clone());
        self.final_page_commitment = Some(prover_data[1].commit.clone());

        tracing::debug!(
            "heights of all traces: {} {} {}",
            self.init_chip_trace.as_ref().unwrap().height(),
            self.offline_checker_trace.as_ref().unwrap().height(),
            self.final_chip_trace.as_ref().unwrap().height()
        );

        (
            vec![
                self.init_chip_trace.clone().unwrap(),
                self.final_chip_trace.clone().unwrap(),
            ],
            prover_data,
        )
    }
}
