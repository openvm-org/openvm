//! Air with columns
//! | count | fields[..] |
//!
//! Chip will either send or receive the fields with multiplicity count.
//! The main Air has no constraints, the only constraints are specified by the Chip trait

use std::{iter, sync::Arc};

use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::{InteractionBuilder, InteractionType},
    prover::types::{AirProofInput, AirProofRawInput, CommittedTraceData, TraceCommitter},
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip,
};
use itertools::izip;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};

pub struct DummyInteractionCols;
impl DummyInteractionCols {
    pub fn count_col() -> usize {
        0
    }
    pub fn field_col(field_idx: usize) -> usize {
        field_idx + 1
    }
}

#[derive(Clone, Copy)]
pub struct DummyInteractionAir {
    field_width: usize,
    /// Send if true. Receive if false.
    pub is_send: bool,
    bus_index: usize,
    /// If true, then | count | and | fields[..] | are in separate main trace partitions.
    pub partition: bool,
}

impl DummyInteractionAir {
    pub fn new(field_width: usize, is_send: bool, bus_index: usize) -> Self {
        Self {
            field_width,
            is_send,
            bus_index,
            partition: false,
        }
    }

    pub fn partition(self) -> Self {
        Self {
            partition: true,
            ..self
        }
    }

    pub fn field_width(&self) -> usize {
        self.field_width
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for DummyInteractionAir {}
impl<F: Field> PartitionedBaseAir<F> for DummyInteractionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        if self.partition {
            vec![self.field_width]
        } else {
            vec![]
        }
    }
    fn common_main_width(&self) -> usize {
        if self.partition {
            1
        } else {
            1 + self.field_width
        }
    }
}
impl<F: Field> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        1 + self.field_width
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: InteractionBuilder + PartitionedAirBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, builder: &mut AB) {
        let (fields, count) = if self.partition {
            let local_0 = builder.common_main().row_slice(0);
            let local_1 = builder.cached_mains()[0].row_slice(0);
            let count = local_0[0];
            let fields = local_1.to_vec();
            (fields, count)
        } else {
            let main = builder.main();
            let local = main.row_slice(0);
            let count = local[DummyInteractionCols::count_col()];
            let fields: Vec<_> = (0..self.field_width)
                .map(|i| local[DummyInteractionCols::field_col(i)])
                .collect();
            (fields, count)
        };
        let interaction_type = if self.is_send {
            InteractionType::Send
        } else {
            InteractionType::Receive
        };
        builder.push_interaction(self.bus_index, fields, count, interaction_type)
    }
}

/// Note: in principle, committing cached trace is out of scope of a chip. But this chip is for
/// usually testing, so we support it for convenience.
pub struct DummyInteractionChip<'a, SC: StarkGenericConfig> {
    trace_committer: Option<TraceCommitter<'a, SC>>,
    // common_main: Option<RowMajorMatrix<Val<SC>>>,
    data: Option<DummyInteractionData>,
    pub air: DummyInteractionAir,
}

#[derive(Debug, Clone)]
pub struct DummyInteractionData {
    pub count: Vec<u32>,
    pub fields: Vec<Vec<u32>>,
}

impl<'a, SC: StarkGenericConfig> DummyInteractionChip<'a, SC>
where
    Val<SC>: AbstractField,
{
    pub fn new_without_partition(field_width: usize, is_send: bool, bus_index: usize) -> Self {
        let air = DummyInteractionAir::new(field_width, is_send, bus_index);
        Self {
            trace_committer: None,
            data: None,
            air,
        }
    }
    pub fn new_with_partition(
        pcs: &'a SC::Pcs,
        field_width: usize,
        is_send: bool,
        bus_index: usize,
    ) -> Self {
        let air = DummyInteractionAir::new(field_width, is_send, bus_index).partition();
        Self {
            trace_committer: Some(TraceCommitter::new(pcs)),
            data: None,
            air,
        }
    }
    pub fn load_data(&mut self, data: DummyInteractionData) {
        let DummyInteractionData { count, fields } = &data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        self.data = Some(data);
    }
    fn generate_traces_with_partition(
        &self,
        data: DummyInteractionData,
    ) -> (RowMajorMatrix<Val<SC>>, CommittedTraceData<SC>) {
        let DummyInteractionData {
            mut count,
            mut fields,
        } = data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        let h = h.next_power_of_two();
        count.resize(h, 0);
        fields.resize(h, vec![0; w]);
        let common_main_val: Vec<_> = count
            .into_iter()
            .map(Val::<SC>::from_canonical_u32)
            .collect();
        let cached_trace_val: Vec<_> = fields
            .into_iter()
            .flatten()
            .map(Val::<SC>::from_canonical_u32)
            .collect();
        let cached_trace = RowMajorMatrix::new(cached_trace_val, w);
        let prover_data = self
            .trace_committer
            .as_ref()
            .unwrap()
            .commit(vec![cached_trace.clone()]);
        (
            RowMajorMatrix::new(common_main_val, 1),
            CommittedTraceData {
                raw_data: cached_trace,
                prover_data,
            },
        )
    }

    fn generate_traces_without_partition(
        &self,
        data: DummyInteractionData,
    ) -> RowMajorMatrix<Val<SC>> {
        let DummyInteractionData { count, fields } = data;
        let h = count.len();
        assert_eq!(fields.len(), h);
        let w = fields[0].len();
        assert_eq!(self.air.field_width, w);
        assert!(fields.iter().all(|r| r.len() == w));
        let common_main_val: Vec<_> = izip!(count, fields)
            .flat_map(|(count, fields)| iter::once(count).chain(fields))
            .chain(iter::repeat(0))
            .take((w + 1) * h.next_power_of_two())
            .map(Val::<SC>::from_canonical_u32)
            .collect();
        RowMajorMatrix::new(common_main_val, w + 1)
    }
}

impl<'a, SC: StarkGenericConfig> Chip<SC> for DummyInteractionChip<'a, SC> {
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(&self) -> AirProofInput<SC> {
        assert!(self.data.is_some());
        let data = self.data.clone().unwrap();
        if self.trace_committer.is_some() {
            let (common_main, cached_main) = self.generate_traces_with_partition(data);
            AirProofInput {
                air: self.air(),
                cached_mains_pdata: vec![cached_main.prover_data],
                raw: AirProofRawInput {
                    cached_mains: vec![Arc::new(cached_main.raw_data)],
                    common_main: Some(common_main),
                    public_values: vec![],
                },
            }
        } else {
            let common_main = self.generate_traces_without_partition(data);
            AirProofInput {
                air: self.air(),
                cached_mains_pdata: vec![],
                raw: AirProofRawInput {
                    cached_mains: vec![],
                    common_main: Some(common_main),
                    public_values: vec![],
                },
            }
        }
    }
}
