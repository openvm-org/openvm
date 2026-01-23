use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    SubAir,
    utils::{assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::{MultiStarkVerifyingKey0V2, MultiStarkVerifyingKeyV2},
    poly_common::Squarable,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqSharpUniBus, EqSharpUniMessage, EqZeroNBus,
        EqZeroNMessage,
    },
    bus::{XiRandomnessBus, XiRandomnessMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        MultiProofVecVec, base_to_ext, ext_field_add, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_one_minus, ext_field_subtract,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqSharpUniCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    root: T,
    root_pow: T,
    root_half_order: T,

    xi_idx: T,
    xi: [T; D_EF],
    product_before: [T; D_EF],

    iter_idx: T,
    is_first_iter: T,
}

pub struct EqSharpUniAir {
    pub xi_bus: XiRandomnessBus,
    pub eq_bus: EqSharpUniBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,
    pub l_skip: usize,
    pub canonical_inverse_generator: F,
}

impl<F> BaseAirWithPublicValues<F> for EqSharpUniAir {}
impl<F> PartitionedBaseAir<F> for EqSharpUniAir {}

impl<F> BaseAir<F> for EqSharpUniAir {
    fn width(&self) -> usize {
        EqSharpUniCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqSharpUniAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
    AB::Expr: From<F>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqSharpUniCols<AB::Var> = (*local).borrow();
        let next: &EqSharpUniCols<AB::Var> = (*next).borrow();

        // Summary:
        // - iter_idx consistency: TODO to grow the nested-loop sub-AIR; the first valid row
        //   initializes `is_first_iter` and `iter_idx`, invalid rows force `iter_idx = 0`, each
        //   step either increments or wraps `iter_idx`, wrapping signals that `root_half_order` was
        //   reached, sets `next.is_first_iter`, and is enforced immediately after the header to
        //   keep layer sizes sound.
        // - Root consistency: continuing iterations multiply `root_pow` by `root` while preserving
        //   `root` and its half order; when a wrap happens, the root squares and the half order
        //   doubles; the first row fixes `root = -1`, `root_half_order = 1`, and `root_pow = 1`,
        //   and the final row forces the root to equal `canonical_inverse_generator`.
        // - Xi/product consistency: the initial product equals one and `xi_idx` starts at `l_skip -
        //   1`, wraps decrement `xi_idx` until reaching zero on the last row, and bus
        //   communications consume and emit products based on `xi`, `1 - xi`, and `xi * root_pow`
        //   combinations.

        type LoopSubAir = NestedForLoopSubAir<1, 0>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_first_iter);

        // =========================== idx consistency =============================
        // TODO: just increase the dimension of nested loop subair?
        builder
            .when(local.is_valid * local.is_first)
            .assert_one(local.is_first_iter);
        builder
            .when(local.is_first_iter)
            .assert_zero(local.iter_idx);
        builder.when(local.is_first_iter).assert_one(local.is_valid);

        // Either iter_idx becomes zero, or increases by one.
        builder.assert_zero(next.iter_idx * (next.iter_idx - local.iter_idx - AB::Expr::ONE));
        // iter_idx is always zero on invalid rows
        builder
            .when(not(local.is_valid))
            .assert_zero(local.iter_idx);
        // If becomes zero, then it would have become root_half_order
        builder
            .when(next.iter_idx - local.iter_idx - AB::Expr::ONE)
            .when(local.is_valid)
            .assert_eq(local.iter_idx + AB::Expr::ONE, local.root_half_order);
        // and, additionally, would set is_first_iter, if next was even valid
        builder
            .when(next.iter_idx - local.iter_idx - AB::Expr::ONE)
            .when(next.is_valid)
            .assert_one(next.is_first_iter);
        // TODO(AG): get rid of iter_idx, use root_pow * root instead (if possible)

        // =========================== Root consistency =============================
        // If iter_idx doesn't become zero (increases by one), then:
        // - root_pow is multiplied by root,
        // - root is preserved.
        builder
            .when(next.iter_idx)
            .assert_eq(next.root_pow, local.root_pow * local.root);
        builder.when(next.iter_idx).assert_eq(local.root, next.root);
        builder
            .when(next.iter_idx)
            .assert_eq(local.root_half_order, next.root_half_order);
        // Otherwise the new root is the square of the current one, and the root order doubles.
        builder
            .when(next.is_first_iter * not(next.is_first))
            .assert_eq(local.root_half_order * AB::Expr::TWO, next.root_half_order);
        builder
            .when(next.is_first_iter * not(next.is_first))
            .assert_eq(local.root, next.root * next.root);
        // Important: we need to enforce dropping iter_idx to zero if root_pow is going to become
        // one. We can leave this to inner interactions, but then we need to guarantee that
        // the first layer has size 1.
        builder
            .when(next.is_valid * local.is_first)
            .assert_one(next.is_first_iter);
        // Also, on the first row we have some conditions
        builder
            .when(local.is_valid * local.is_first)
            .assert_eq(local.root, AB::Expr::NEG_ONE);
        builder
            .when(local.is_valid * local.is_first)
            .assert_eq(local.root_half_order, AB::Expr::ONE);
        // If is_first_iter, then root_pow is 1
        builder
            .when(local.is_first_iter)
            .assert_eq(local.root_pow, AB::Expr::ONE);
        // Finally, the final root must equal some specific generator
        builder
            .when(local.is_valid * local.is_last)
            .assert_eq::<AB::Var, AB::Expr>(local.root, self.canonical_inverse_generator.into());

        // =========================== Xi and product consistency =============================
        // Boundary conditions
        assert_array_eq(
            &mut builder.when(local.is_valid * local.is_first),
            local.product_before,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        builder.when(local.is_valid * local.is_first).assert_eq(
            local.xi_idx,
            AB::Expr::from_canonical_usize(self.l_skip) - AB::Expr::ONE,
        );
        // When we drop iter_idx, xi_idx decreases
        builder
            .when(next.is_first_iter * not(next.is_first))
            .assert_one(local.xi_idx - next.xi_idx);
        // The last one must be 0
        builder
            .when(local.is_last)
            .assert_eq(local.xi_idx, AB::Expr::ZERO);

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.xi_idx,
                xi: local.xi,
            },
            local.is_first_iter,
        );
        self.batch_constraint_conductor_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.xi_idx.into(),
                value: local.xi.map(|x| x.into()),
            },
            local.is_valid * local.is_last,
        );

        self.eq_bus.receive(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: local.xi_idx + AB::Expr::ONE,
                iter_idx: local.iter_idx.into(),
                product: local.product_before.map(|x| x.into()),
            },
            local.is_valid * (AB::Expr::ONE - local.is_first),
        );
        let second: [AB::Expr; D_EF] = ext_field_multiply_scalar(local.xi, local.root_pow);
        let one_minus_xi: [AB::Expr; D_EF] = ext_field_one_minus(local.xi);

        self.eq_bus.send(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: local.xi_idx.into(),
                iter_idx: local.iter_idx.into(),
                product: ext_field_multiply(
                    local.product_before,
                    ext_field_add::<AB::Expr>(one_minus_xi.clone(), second.clone()),
                ),
            },
            local.is_valid,
        );
        self.eq_bus.send(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: local.xi_idx.into(),
                iter_idx: local.iter_idx + local.root_half_order,
                product: ext_field_multiply(
                    local.product_before,
                    ext_field_subtract::<AB::Expr>(one_minus_xi.clone(), second.clone()),
                ),
            },
            local.is_valid,
        );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqSharpUniReceiverCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    idx: T,
    coeff: [T; D_EF],
    r: [T; D_EF],
    cur_sum: [T; D_EF],
}

pub struct EqSharpUniReceiverAir {
    pub r_bus: BatchConstraintConductorBus,
    pub eq_bus: EqSharpUniBus,
    pub zero_n_bus: EqZeroNBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqSharpUniReceiverAir {}
impl<F> PartitionedBaseAir<F> for EqSharpUniReceiverAir {}

impl<F> BaseAir<F> for EqSharpUniReceiverAir {
    fn width(&self) -> usize {
        EqSharpUniReceiverCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqSharpUniReceiverAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqSharpUniReceiverCols<AB::Var> = (*local).borrow();
        let next: &EqSharpUniReceiverCols<AB::Var> = (*next).borrow();

        // Summary:
        // - idx consistency: start with `idx = 0` on the first row and increment by one on each
        //   subsequent non-header row while the proof continues.
        // - EF values consistency: keep the `r` coefficients constant across transitions and update
        //   `cur_sum` via `coeff + r * next.cur_sum` on every step past the first.

        type LoopSubAir = NestedForLoopSubAir<1, 0>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);

        // ============================= idx consistency ============================
        builder.when(local.is_first).assert_zero(local.idx);
        builder
            .when(not(next.is_first))
            .assert_one(next.idx - local.idx);
        // ============================= EF values consistency ==========================
        assert_array_eq(&mut builder.when(not(next.is_first)), next.r, local.r);
        assert_array_eq(
            &mut builder.when(not(next.is_first)),
            local.cur_sum,
            ext_field_add(local.coeff, ext_field_multiply(local.r, next.cur_sum)),
        );

        self.r_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: AB::Expr::ZERO,
                value: local.r.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );
        self.eq_bus.receive(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: AB::Expr::ZERO,
                iter_idx: local.idx.into(),
                product: local.coeff.map(|x| x.into()),
            },
            local.is_valid,
        );
        let l_skip_inv: AB::Expr = AB::F::from_canonical_usize(1 << self.l_skip)
            .inverse()
            .into();
        self.zero_n_bus.send(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ONE,
                value: local.cur_sum.map(|x| l_skip_inv.clone() * x),
            },
            local.is_valid * local.is_first,
        );
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EqSharpUniRecord {
    xi: EF,
    product: EF,

    root: F,
    root_pow: F,

    // PERF: we can only store `xi_idx: u32` and later take `xi` from the transcript
    xi_idx: u32,
    root_half_order: u32,
    iter_idx: u32,
}

#[derive(Debug, Clone)]
pub struct EqSharpUniBlob {
    records: MultiProofVecVec<EqSharpUniRecord>,
    final_products: MultiProofVecVec<EF>,
    rs: Vec<EF>,
}

impl EqSharpUniBlob {
    fn new(l_skip: usize, num_proofs: usize) -> Self {
        Self {
            records: MultiProofVecVec::with_capacity(num_proofs * ((1 << l_skip) - 1)),
            final_products: MultiProofVecVec::with_capacity(num_proofs << l_skip),
            rs: Vec::with_capacity(num_proofs),
        }
    }
}

pub fn generate_eq_sharp_uni_blob(
    vk: &MultiStarkVerifyingKey0V2,
    preflights: &[&Preflight],
) -> EqSharpUniBlob {
    let l_skip = vk.params.l_skip;
    let mut blob = EqSharpUniBlob::new(l_skip, preflights.len());
    let mut products = vec![EF::ONE; 1 << l_skip];
    let roots = F::two_adic_generator(l_skip)
        .inverse()
        .exp_powers_of_2()
        .take(l_skip)
        .collect::<Vec<_>>();
    for preflight in preflights.iter() {
        products[0] = EF::ONE;
        for i in 0..l_skip {
            let xi_idx = l_skip - 1 - i;
            let xi = preflight.gkr.xi[xi_idx].1;
            let root = roots[l_skip - 1 - i];
            let mut root_pow = F::ONE;
            for (j, &product) in products.iter().take(1 << i).enumerate() {
                blob.records.push(EqSharpUniRecord {
                    xi,
                    product,
                    root,
                    root_pow,
                    xi_idx: xi_idx as u32,
                    root_half_order: 1 << i,
                    iter_idx: j as u32,
                });
                root_pow *= root;
            }
            for j in (0..(1 << i)).rev() {
                let value = products[j];
                let root_pow = blob.records.data()[blob.records.len() - (1 << i) + j].root_pow;
                let second = xi * root_pow;
                products[j + (1 << i)] = value * (EF::ONE - xi - second);
                products[j] = value * (EF::ONE - xi + second);
            }
        }
        blob.records.end_proof();
        blob.final_products.extend_from_slice(&products);
        blob.final_products.end_proof();
        blob.rs.push(preflight.batch_constraint.sumcheck_rnd[0]);
    }
    blob
}

#[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
pub(crate) fn generate_eq_sharp_uni_trace(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &EqSharpUniBlob,
    preflights: &[&Preflight],
) -> RowMajorMatrix<F> {
    let width = EqSharpUniCols::<F>::width();
    let l_skip = vk.inner.params.l_skip;
    let one_height = (1 << l_skip) - 1;
    let total_height = one_height * preflights.len();

    let padded_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

    for pidx in 0..preflights.len() {
        let records = &blob.records[pidx];
        trace[(pidx * one_height * width)..((pidx + 1) * one_height * width)]
            .par_chunks_exact_mut(width)
            .zip(records.par_iter())
            .enumerate()
            .for_each(|(i, (chunk, record))| {
                let cols: &mut EqSharpUniCols<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == one_height);
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.is_first_iter = F::ONE;
                cols.xi_idx = F::from_canonical_u32(record.xi_idx);
                cols.xi.copy_from_slice(record.xi.as_base_slice());
                cols.iter_idx = F::from_canonical_u32(record.iter_idx);
                cols.is_first_iter = F::from_bool(record.iter_idx == 0);
                cols.product_before
                    .copy_from_slice(record.product.as_base_slice());
                cols.root = record.root;
                cols.root_pow = record.root_pow;
                cols.root_half_order = F::from_canonical_u32(record.root_half_order);
            });
    }

    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqSharpUniCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}

#[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
pub(crate) fn generate_eq_sharp_uni_receiver_trace(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &EqSharpUniBlob,
    preflights: &[&Preflight],
) -> RowMajorMatrix<F> {
    let l_skip = vk.inner.params.l_skip;

    let width = EqSharpUniReceiverCols::<F>::width();
    let one_height = 1 << l_skip;
    let total_height = one_height * preflights.len();
    let padded_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

    for pidx in 0..preflights.len() {
        let products = &blob.final_products[pidx];
        let r = blob.rs[pidx];
        trace[(pidx * one_height * width)..((pidx + 1) * one_height * width)]
            .par_chunks_exact_mut(width)
            .zip(products.par_iter())
            .enumerate()
            .for_each(|(i, (chunk, product))| {
                let cols: &mut EqSharpUniReceiverCols<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == one_height);
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.coeff.copy_from_slice(product.as_base_slice());
                cols.r.copy_from_slice(r.as_base_slice());
                cols.idx = F::from_canonical_usize(i);
            });
        let mut cur_sum = EF::ZERO;
        trace[(pidx * one_height * width)..((pidx + 1) * one_height * width)]
            .chunks_exact_mut(width)
            .rev()
            .for_each(|chunk| {
                let cols: &mut EqSharpUniReceiverCols<_> = chunk.borrow_mut();
                cur_sum = cur_sum * r + EF::from_base_slice(&cols.coeff);
                cols.cur_sum.copy_from_slice(cur_sum.as_base_slice());
            });
    }

    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqSharpUniReceiverCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
