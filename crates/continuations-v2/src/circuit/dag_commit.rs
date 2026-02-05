use core::borrow::Borrow;
use std::{array::from_fn, borrow::BorrowMut, sync::Arc};

use itertools::Itertools;
use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_poseidon2_air::{
    BABY_BEAR_POSEIDON2_SBOX_DEGREE, POSEIDON2_WIDTH, Poseidon2Config, Poseidon2SubAir,
    Poseidon2SubChip, Poseidon2SubCols,
};
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, InjectiveMonomial, PrimeCharacteristicRing, PrimeField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use recursion_circuit::{
    batch_constraint::expr_eval::cached_symbolic_expr_cols_to_digest,
    bus::{DagCommitBus, DagCommitBusMessage},
    utils::assert_zeros,
};
use stark_backend_v2::{
    DIGEST_SIZE,
    prover::{ColMajorMatrix, DeviceDataTransporterV2, ProverBackendV2, StridedColMajorMatrixView},
};
use stark_recursion_circuit_derive::AlignedBorrow;

const SBOX_REGISTERS: usize = 1;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DagCommitCols<T> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub row_idx: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DagCommitPvs<T> {
    pub commit: [T; DIGEST_SIZE],
}

// TODO[INT-5923]: Make DagCommitAir a sub-AIR in SymbolicExpressionAir
pub struct DagCommitAir<F: Field> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub dag_commit_bus: DagCommitBus,
}

impl<F: PrimeField + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>> DagCommitAir<F> {
    pub fn new(dag_commit_bus: DagCommitBus) -> Self {
        let sub_chip =
            Poseidon2SubChip::<F, SBOX_REGISTERS>::new(Poseidon2Config::default().constants);
        Self {
            subair: sub_chip.air,
            dag_commit_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for DagCommitAir<F> {
    fn width(&self) -> usize {
        DagCommitCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DagCommitAir<F> {
    fn num_public_values(&self) -> usize {
        DagCommitPvs::<F>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for DagCommitAir<F> {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for DagCommitAir<AB::F>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &DagCommitCols<AB::Var> = (*local).borrow();
        let next: &DagCommitCols<AB::Var> = (*next).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_one(next.row_idx - local.row_idx);

        assert_zeros::<_, DIGEST_SIZE>(
            &mut builder.when_first_row(),
            from_fn(|i| local.inner.inputs[i + DIGEST_SIZE]),
        );
        assert_array_eq::<_, _, _, DIGEST_SIZE>(
            &mut builder.when_transition(),
            from_fn(|i| local.inner.ending_full_rounds.last().unwrap().post[i + DIGEST_SIZE]),
            from_fn(|i| next.inner.inputs[i + DIGEST_SIZE]),
        );

        self.dag_commit_bus.receive(
            builder,
            AB::Expr::ZERO,
            DagCommitBusMessage {
                idx: local.row_idx,
                values: from_fn(|i| local.inner.inputs[i]),
            },
            AB::Expr::ONE,
        );

        let &DagCommitPvs::<_> { commit: pvs_commit } = builder.public_values().borrow();

        assert_array_eq::<_, _, _, DIGEST_SIZE>(
            &mut builder.when_last_row(),
            from_fn(|i| local.inner.ending_full_rounds.last().unwrap().post[i]),
            pvs_commit,
        );
    }
}

#[tracing::instrument(name = "generate_cached_trace", skip_all)]
pub fn generate_dag_commit_proving_ctx<PB: ProverBackendV2>(
    device: impl DeviceDataTransporterV2<PB>,
    cached_trace: PB::Matrix,
) -> (PB::Matrix, [PB::Val; DIGEST_SIZE])
where
    PB::Val: Field + PrimeField + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
{
    let host_trace_cm = device.transport_matrix_from_device_to_host(&cached_trace);
    let host_trace_view: StridedColMajorMatrixView<_> = host_trace_cm.as_view().into();
    let host_trace = host_trace_view.to_row_major_matrix();

    let sub_chip =
        Poseidon2SubChip::<PB::Val, SBOX_REGISTERS>::new(Poseidon2Config::default().constants);
    let mut state = [PB::Val::ZERO; POSEIDON2_WIDTH];

    debug_assert!(host_trace.values.len() % DIGEST_SIZE == 0);
    let mut inputs = host_trace
        .row_slices()
        .map(|row| {
            state[..DIGEST_SIZE].copy_from_slice(&cached_symbolic_expr_cols_to_digest(row));
            let input = state.clone();
            sub_chip.permute_mut(&mut state);
            input
        })
        .collect_vec();

    let height = inputs.len().next_power_of_two();
    let width = DagCommitCols::<PB::Val>::width();
    let mut trace_values = vec![PB::Val::ZERO; height * width];
    let mut trace_iter = trace_values.chunks_mut(width);

    inputs.resize(height, [PB::Val::ZERO; POSEIDON2_WIDTH]);
    let inner_trace = sub_chip.generate_trace(inputs);
    let inner_width = sub_chip.air.width();

    for (row_idx, inner_row) in inner_trace.row_slices().enumerate() {
        let row = trace_iter.next().unwrap();
        row[..inner_width].copy_from_slice(inner_row);
        let cols: &mut DagCommitCols<PB::Val> = row.borrow_mut();
        cols.row_idx = PB::Val::from_usize(row_idx);
    }

    let matrix_rm = RowMajorMatrix::new(trace_values, width);
    let matrix = ColMajorMatrix::from_row_major(&matrix_rm);
    let device_trace = device.transport_matrix_to_device(&matrix);
    (device_trace, from_fn(|i| state[i]))
}
