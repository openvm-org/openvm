//! Halo2 implementation of poseidon2 perm for Bn254
//! sbox degree 5

use halo2_base::{
    gates::{range::RangeChip, GateInstructions, RangeInstructions},
    safe_types::SafeBool,
    utils::ScalarField,
    AssignedValue, Context,
    QuantumCell::{self, Constant},
};
pub(crate) use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::{
    BABY_BEAR_RATE as MULTI_FIELD32_RATE, BN254_RATE as POSEIDON2_RATE, DIGEST_WIDTH,
};

use crate::{field::baby_bear::BabyBearWire, Fr};

const MULTI_FIELD32_NUM_F_ELMS: usize = MULTI_FIELD32_RATE / POSEIDON2_RATE;

#[derive(Clone, Debug)]
pub struct Poseidon2State<F: ScalarField, const T: usize> {
    pub s: [AssignedValue<F>; T],
}

#[derive(Debug, Clone)]
pub struct Poseidon2Params<F: ScalarField, const T: usize> {
    /// Number of full rounds
    pub rounds_f: usize,
    pub rounds_p: usize,
    pub mat_internal_diag_m_1: [F; T],
    pub external_rc: Vec<[F; T]>,
    pub internal_rc: Vec<F>,
}

impl<F: ScalarField, const T: usize> Poseidon2Params<F, T> {
    pub fn new(
        rounds_f: usize,
        rounds_p: usize,
        mat_internal_diag_m_1: [F; T],
        external_rc: Vec<[F; T]>,
        internal_rc: Vec<F>,
    ) -> Self {
        Self {
            rounds_f,
            rounds_p,
            mat_internal_diag_m_1,
            external_rc,
            internal_rc,
        }
    }
}

impl<F: ScalarField, const T: usize> Poseidon2State<F, T> {
    pub fn new(state: [AssignedValue<F>; T]) -> Self {
        Self { s: state }
    }
    /// Perform permutation on this state.
    ///
    /// ATTENTION: inputs.len() needs to be fixed at compile time.
    pub fn permutation(
        &mut self,
        ctx: &mut Context<F>,
        gate: &impl GateInstructions<F>,
        params: &Poseidon2Params<F, T>,
    ) {
        let rounds_f_beginning = params.rounds_f / 2;

        // First half of the full round
        self.matmul_external(ctx, gate);
        for r in 0..rounds_f_beginning {
            self.add_rc(ctx, gate, params.external_rc[r]);
            self.sbox(ctx, gate);
            self.matmul_external(ctx, gate);
        }

        for r in 0..params.rounds_p {
            self.s[0] = gate.add(ctx, self.s[0], Constant(params.internal_rc[r]));
            self.s[0] = Self::x_power5(ctx, gate, self.s[0]);
            self.matmul_internal(ctx, gate, params.mat_internal_diag_m_1);
        }

        for r in rounds_f_beginning..params.rounds_f {
            self.add_rc(ctx, gate, params.external_rc[r]);
            self.sbox(ctx, gate);
            self.matmul_external(ctx, gate);
        }
    }

    /// Constrains and set self to a specific state if `selector` is true.
    pub fn select(
        &mut self,
        ctx: &mut Context<F>,
        gate: &impl GateInstructions<F>,
        selector: SafeBool<F>,
        set_to: &Self,
    ) {
        for i in 0..T {
            self.s[i] = gate.select(ctx, set_to.s[i], self.s[i], *selector.as_ref());
        }
    }

    fn x_power5(
        ctx: &mut Context<F>,
        gate: &impl GateInstructions<F>,
        x: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let x2 = gate.mul(ctx, x, x);
        let x4 = gate.mul(ctx, x2, x2);
        gate.mul(ctx, x, x4)
    }

    fn sbox(&mut self, ctx: &mut Context<F>, gate: &impl GateInstructions<F>) {
        for x in self.s.iter_mut() {
            *x = Self::x_power5(ctx, gate, *x);
        }
    }

    fn matmul_external(&mut self, ctx: &mut Context<F>, gate: &impl GateInstructions<F>) {
        // Only doing T = 3 case
        assert_eq!(T, 3);

        // Matrix is circ(2, 1, 1)
        let sum = gate.sum(ctx, self.s.iter().copied());
        for (i, x) in self.s.iter_mut().enumerate() {
            // This is the same as `*x = gate.add(ctx, *x, sum)` but we save a cell by reusing
            // `sum`:
            if i % 2 == 0 {
                ctx.assign_region(
                    [
                        QuantumCell::Witness(*x.value() + sum.value()),
                        QuantumCell::Existing(*x),
                        QuantumCell::Constant(-F::ONE),
                        QuantumCell::Existing(sum),
                    ],
                    [0],
                );
                *x = ctx.get(-4);
            } else {
                ctx.assign_region(
                    [
                        QuantumCell::Existing(*x),
                        QuantumCell::Constant(F::ONE),
                        QuantumCell::Witness(*x.value() + sum.value()),
                    ],
                    [-1],
                );
                *x = ctx.get(-1);
            }
        }
    }

    fn add_rc(
        &mut self,
        ctx: &mut Context<F>,
        gate: &impl GateInstructions<F>,
        round_constants: [F; T],
    ) {
        for (x, rc) in self.s.iter_mut().zip(round_constants.iter()) {
            *x = gate.add(ctx, *x, Constant(*rc));
        }
    }

    fn matmul_internal(
        &mut self,
        ctx: &mut Context<F>,
        gate: &impl GateInstructions<F>,
        mat_internal_diag_m_1: [F; T],
    ) {
        assert_eq!(T, 3);
        let sum = gate.sum(ctx, self.s.iter().copied());
        #[allow(clippy::needless_range_loop)]
        for i in 0..T {
            // This is the same as `self.s[i] = gate.mul_add(ctx, self.s[i],
            // Constant(mat_internal_diag_m_1[i]), sum)` but we save a cell by reusing `sum`.
            if i % 2 == 0 {
                ctx.assign_region(
                    [
                        QuantumCell::Witness(
                            *self.s[i].value() * mat_internal_diag_m_1[i] + sum.value(),
                        ),
                        QuantumCell::Existing(self.s[i]),
                        QuantumCell::Constant(-mat_internal_diag_m_1[i]),
                        QuantumCell::Existing(sum),
                    ],
                    [0],
                );
                self.s[i] = ctx.get(-4);
            } else {
                ctx.assign_region(
                    [
                        QuantumCell::Existing(self.s[i]),
                        QuantumCell::Constant(mat_internal_diag_m_1[i]),
                        QuantumCell::Witness(
                            *self.s[i].value() * mat_internal_diag_m_1[i] + sum.value(),
                        ),
                    ],
                    [-1],
                );
                self.s[i] = ctx.get(-1);
            }
        }
    }
}

pub(crate) fn reduce_32_cells(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    values: &[AssignedValue<Fr>],
) -> AssignedValue<Fr> {
    let base = Fr::from(1u64 << 32);
    gate.inner_product(
        ctx,
        values.iter().copied(),
        core::iter::successors(Some(Fr::from(1u64)), |power| Some(*power * base))
            .take(values.len())
            .map(Constant),
    )
}

pub(crate) fn hash_babybear_slice_to_digest(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    values: &[BabyBearWire],
) -> AssignedValue<Fr> {
    let gate = range.gate();
    let params = &*super::POSEIDON2_PARAMS;
    let zero = ctx.load_zero();
    let mut state = Poseidon2State::new(core::array::from_fn(|_| zero));
    for block_chunk in values.chunks(MULTI_FIELD32_RATE) {
        for (chunk_id, chunk) in block_chunk.chunks(MULTI_FIELD32_NUM_F_ELMS).enumerate() {
            let cells = chunk.iter().map(|value| value.value).collect::<Vec<_>>();
            state.s[chunk_id] = reduce_32_cells(ctx, gate, &cells);
        }
        state.permutation(ctx, gate, params);
    }
    state.s[0]
}

pub(crate) fn compress_bn254_digests(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    left: AssignedValue<Fr>,
    right: AssignedValue<Fr>,
) -> AssignedValue<Fr> {
    let gate = range.gate();
    let params = &*super::POSEIDON2_PARAMS;
    let zero = ctx.load_zero();
    let mut state = Poseidon2State::new([left, right, zero]);
    state.permutation(ctx, gate, params);
    state.s[0]
}
