use crate::group_by::final_page::MyFinalPageAir;
use crate::group_by::group_by_input::GroupByAir;
use crate::range_gate::RangeCheckerGateChip;
use afs_stark_backend::config::Com;
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_stark_backend::prover::trace::TraceCommitter;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use std::sync::Arc;

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField,
{
    pub group_by: GroupByAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub final_chip: MyFinalPageAir,

    group_by_trace: Option<DenseMatrix<Val<SC>>>,
    final_page_trace: Option<DenseMatrix<Val<SC>>>,
    final_page_aux_trace: Option<DenseMatrix<Val<SC>>>,

    group_by_commitment: Option<Com<SC>>,
    final_page_commitment: Option<Com<SC>>,
}

impl<SC: StarkGenericConfig> PageController<SC> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
        range_bus: usize,
        limb_bits: usize,
        decomp: usize,
    ) -> Self {
        let group_by = GroupByAir::new(
            page_width,
            group_by_cols.clone(),
            aggregated_col,
            internal_bus,
            output_bus,
        );
        let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus, 1 << decomp));
        let final_chip = MyFinalPageAir::new(
            output_bus,
            range_bus,
            group_by_cols.len(),
            1,
            limb_bits,
            decomp,
        );
        Self {
            group_by,
            range_checker,
            final_chip,
            group_by_trace: None,
            final_page_trace: None,
            final_page_aux_trace: None,
            group_by_commitment: None,
            final_page_commitment: None,
        }
    }

    pub fn load_page(
        &mut self,
        page: Vec<Vec<u32>>,
        trace_committer: &TraceCommitter<SC>,
    ) -> (Vec<DenseMatrix<Val<SC>>>, Vec<ProverTraceData<SC>>)
    where
        Val<SC>: PrimeField,
    {
        self.group_by_trace = Some(self.group_by.gen_page_trace(page.clone()));

        let mut grouped_page: Vec<Vec<u32>> = page
            .iter()
            .map(|row| {
                let mut selected_row: Vec<u32> = self
                    .group_by
                    .group_by_cols
                    .iter()
                    .map(|&col_index| row[col_index])
                    .collect();
                selected_row.push(row[self.group_by.aggregated_col]);
                selected_row
            })
            .collect();

        grouped_page.sort();

        let mut sums_by_key: std::collections::HashMap<Vec<u32>, u32> =
            std::collections::HashMap::new();
        for row in grouped_page.iter() {
            let (value, index) = row.split_last().unwrap();
            *sums_by_key.entry(index.to_vec()).or_insert(0) += value;
        }
        // Convert the hashmap back to a sorted vector for further processing
        let mut grouped_sums: Vec<Vec<u32>> = sums_by_key
            .into_iter()
            .map(|(mut key, sum)| {
                key.insert(0, 1);
                key.push(sum);
                key
            })
            .collect();
        grouped_sums.sort();

        self.final_page_trace = Some(self.final_chip.gen_page_trace::<SC>(grouped_sums.clone()));
        self.final_page_aux_trace = Some(
            self.final_chip
                .gen_aux_trace::<SC>(grouped_sums, self.range_checker.clone()),
        );

        let prover_data = vec![
            trace_committer.commit(vec![self.group_by_trace.clone().unwrap()]),
            trace_committer.commit(vec![self.final_page_trace.clone().unwrap()]),
        ];

        self.group_by_commitment = Some(prover_data[0].commit.clone());
        self.final_page_commitment = Some(prover_data[1].commit.clone());

        (
            vec![
                self.group_by_trace.clone().unwrap(),
                self.final_page_trace.clone().unwrap(),
            ],
            prover_data,
        )
    }
}
