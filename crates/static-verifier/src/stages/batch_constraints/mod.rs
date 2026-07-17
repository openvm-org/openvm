use std::{collections::BTreeMap, iter::zip};

use halo2_base::AssignedValue;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{
        air_builders::symbolic::{
            symbolic_variable::{Entry, SymbolicVariable},
            SymbolicExpressionNode,
        },
        calculate_n_logup,
        keygen::types::MultiStarkVerifyingKey0,
        p3_field::{Field, PrimeCharacteristicRing, TwoAdicField},
    },
};

use crate::{
    chip_traits::{BabyBearExt4Inst, PopulateInputs, TranscriptInst},
    field::baby_bear::{
        BabyBearExtWire, BabyBearWire, ReducedBabyBearExtWire, ReducedBabyBearWire,
    },
    profiling::CellProfiler,
    stages::shared_math::{
        column_openings_by_rot_assigned, horner_eval_ext_poly_assigned, ExtWirePair,
    },
    Fr, RootEF, RootF,
};

#[derive(Clone, Debug)]
pub struct BatchConstraintIntermediatesWire<F = AssignedValue<Fr>> {
    pub column_openings: Vec<Vec<Vec<ReducedBabyBearExtWire<F>>>>,
    pub r: Vec<BabyBearExtWire<F>>,
}

#[derive(Clone, Debug)]
pub struct GkrProofWire<F = AssignedValue<Fr>> {
    pub logup_pow_witness: ReducedBabyBearWire<F>,
    pub q0_claim: ReducedBabyBearExtWire<F>,
    pub claims_per_layer: Vec<[ReducedBabyBearExtWire<F>; 4]>,
    pub sumcheck_polys: Vec<Vec<[ReducedBabyBearExtWire<F>; 3]>>,
}

#[derive(Clone, Debug)]
pub struct BatchConstraintProofWire<F = AssignedValue<Fr>> {
    pub numerator_term_per_air: Vec<ReducedBabyBearExtWire<F>>,
    pub denominator_term_per_air: Vec<ReducedBabyBearExtWire<F>>,
    pub univariate_round_coeffs: Vec<ReducedBabyBearExtWire<F>>,
    pub sumcheck_round_polys: Vec<Vec<ReducedBabyBearExtWire<F>>>,
    pub column_openings: Vec<Vec<Vec<ReducedBabyBearExtWire<F>>>>,
}

pub(crate) fn load_gkr_proof_wire<B: PopulateInputs>(
    b: &mut B,
    gkr_proof: &openvm_stark_sdk::openvm_stark_backend::proof::GkrProof<RootConfig>,
) -> GkrProofWire<B::F> {
    let logup_pow_witness = b.bb_load_reduced_witness(gkr_proof.logup_pow_witness);
    let q0_claim = b.ext_load_reduced_witness(gkr_proof.q0_claim);
    let claims_per_layer = gkr_proof
        .claims_per_layer
        .iter()
        .map(|claims| {
            [
                b.ext_load_reduced_witness(claims.p_xi_0),
                b.ext_load_reduced_witness(claims.q_xi_0),
                b.ext_load_reduced_witness(claims.p_xi_1),
                b.ext_load_reduced_witness(claims.q_xi_1),
            ]
        })
        .collect::<Vec<_>>();
    let sumcheck_polys = gkr_proof
        .sumcheck_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|evals| evals.map(|value| b.ext_load_reduced_witness(value)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    GkrProofWire {
        logup_pow_witness,
        q0_claim,
        claims_per_layer,
        sumcheck_polys,
    }
}

pub(crate) fn load_batch_constraint_proof_wire<B: PopulateInputs>(
    b: &mut B,
    batch_proof: &openvm_stark_sdk::openvm_stark_backend::proof::BatchConstraintProof<RootConfig>,
) -> BatchConstraintProofWire<B::F> {
    let numerator_term_per_air = batch_proof
        .numerator_term_per_air
        .iter()
        .map(|&value| b.ext_load_reduced_witness(value))
        .collect::<Vec<_>>();
    let denominator_term_per_air = batch_proof
        .denominator_term_per_air
        .iter()
        .map(|&value| b.ext_load_reduced_witness(value))
        .collect::<Vec<_>>();
    let univariate_round_coeffs = batch_proof
        .univariate_round_coeffs
        .iter()
        .map(|&value| b.ext_load_reduced_witness(value))
        .collect::<Vec<_>>();
    let sumcheck_round_polys = batch_proof
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| b.ext_load_reduced_witness(value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let column_openings = batch_proof
        .column_openings
        .iter()
        .map(|per_air| {
            per_air
                .iter()
                .map(|part| {
                    part.iter()
                        .map(|&value| b.ext_load_reduced_witness(value))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    BatchConstraintProofWire {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
    }
}

fn eval_lagrange_on_integer_grid<B: BabyBearExt4Inst>(
    b: &mut B,
    point: &BabyBearExtWire<B::F>,
    evals: &[BabyBearExtWire<B::F>],
) -> BabyBearExtWire<B::F> {
    let n = evals.len().saturating_sub(1);
    let one = b.ext_from_base_const(RootF::ONE);
    let x_grid = (0..=n)
        .map(|j| b.bb_load_constant(RootF::from_u64(j as u64)))
        .collect::<Vec<_>>();
    let mut acc = b.ext_zero();
    for (i, eval_i) in evals.iter().enumerate() {
        let mut basis = one;
        let mut denom = RootF::ONE;
        #[allow(clippy::needless_range_loop)]
        for j in 0..=n {
            if i == j {
                continue;
            }
            let mut x_minus_j = *point;
            x_minus_j.0[0] = b.bb_sub(x_minus_j.0[0], x_grid[j]);
            basis = b.ext_mul(basis, x_minus_j);

            let diff = if i >= j {
                RootF::from_usize(i - j)
            } else {
                -RootF::from_usize(j - i)
            };
            denom *= diff;
        }
        let denom_inv = denom.inverse();
        let basis = b.ext_mul_base_const(basis, denom_inv);
        let term = b.ext_mul(*eval_i, basis);
        acc = b.ext_add(acc, term);
    }
    acc
}

fn progression_exp_2_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    m: &BabyBearExtWire<B::F>,
    l: usize,
) -> BabyBearExtWire<B::F> {
    let mut pow = *m;
    let one = b.ext_from_base_const(RootF::ONE);
    let mut sum = one;
    for _ in 0..l {
        let one_plus_pow = b.ext_add(one, pow);
        sum = b.ext_mul(sum, one_plus_pow);
        pow = b.ext_mul(pow, pow);
    }
    sum
}

pub(crate) fn eval_eq_mle_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    x: &[BabyBearExtWire<B::F>],
    y: &[BabyBearExtWire<B::F>],
) -> BabyBearExtWire<B::F> {
    assert_eq!(x.len(), y.len(), "eq_mle vector length mismatch");
    let one = b.ext_from_base_const(RootF::ONE);
    let mut acc = one;
    // Rewrite: 2xy - x + (1-y) = (1-y) + x(2y-1).
    // This replaces one ext×ext mul (xy) + scalar_mul (2*xy)
    // with just a scalar_mul (2*y) + one ext×ext mul (x * (2y-1)).
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        let two_y_minus_one = b.ext_mul_base_const(*y_i, RootF::TWO);
        let two_y_minus_one = b.ext_sub(two_y_minus_one, one);
        let x_term = b.ext_mul(*x_i, two_y_minus_one);
        let one_minus_y = b.ext_sub(one, *y_i);
        let factor = b.ext_add(one_minus_y, x_term);
        acc = b.ext_mul(acc, factor);
    }
    acc
}

pub(crate) fn eval_eq_mle_ef_f_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    x: &[BabyBearExtWire<B::F>],
    y: &[BabyBearWire<B::F>],
) -> BabyBearExtWire<B::F> {
    assert_eq!(x.len(), y.len(), "eq_mle vector length mismatch");
    let one_base = b.bb_one();
    let mut acc = b.ext_from_base_const(RootF::ONE);
    // Rewrite: 2xy - x + (1-y) = (1-y) + x(2y-1).
    // Since y is base-field: compute 2y-1 as a base-field constant,
    // then scalar_mul x by it. Saves one scalar_mul (the old xy + 2*xy chain).
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        let two_y = b.bb_mul_const(*y_i, RootF::TWO);
        let two_y_minus_one = b.bb_sub(two_y, one_base);
        let x_term = b.ext_scalar_mul(*x_i, two_y_minus_one);
        let one_minus_y = b.bb_sub(one_base, *y_i);
        let mut factor = x_term;
        factor.0[0] = b.bb_add(factor.0[0], one_minus_y);
        acc = b.ext_mul(acc, factor);
    }
    acc
}

pub(crate) fn eval_eq_mle_binary_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    x: &[BabyBearExtWire<B::F>],
    y_bits: &[bool],
) -> BabyBearExtWire<B::F> {
    assert_eq!(
        x.len(),
        y_bits.len(),
        "eq_mle binary vector length mismatch",
    );
    let one = b.ext_from_base_const(RootF::ONE);
    let mut acc = one;
    for (x_i, bit) in x.iter().zip(y_bits.iter().copied()) {
        let factor = if bit { *x_i } else { b.ext_sub(one, *x_i) };
        acc = b.ext_mul(acc, factor);
    }
    acc
}

pub(crate) fn eval_eq_uni_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    l_skip: usize,
    x: &BabyBearExtWire<B::F>,
    y: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let one = b.ext_from_base_const(RootF::ONE);
    let mut res = one;
    let mut x_pow = *x;
    let mut y_pow = *y;
    for _ in 0..l_skip {
        let x_plus_y = b.ext_add(x_pow, y_pow);
        let x_minus_one = b.ext_sub(x_pow, one);
        let y_minus_one = b.ext_sub(y_pow, one);
        let correction = b.ext_mul(x_minus_one, y_minus_one);
        let scaled_res = b.ext_mul(x_plus_y, res);
        res = b.ext_add(scaled_res, correction);
        x_pow = b.ext_mul(x_pow, x_pow);
        y_pow = b.ext_mul(y_pow, y_pow);
    }
    let half_pow_l = RootF::ONE.halve().exp_u64(l_skip as u64);
    b.ext_mul_base_const(res, half_pow_l)
}

pub(crate) fn eval_eq_uni_at_one_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    l_skip: usize,
    x: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let one = b.ext_from_base_const(RootF::ONE);
    let mut res = one;
    let mut x_pow = *x;
    for _ in 0..l_skip {
        let x_plus_one = b.ext_add(x_pow, one);
        res = b.ext_mul(res, x_plus_one);
        x_pow = b.ext_mul(x_pow, x_pow);
    }
    let half_pow_l = RootF::ONE.halve().exp_u64(l_skip as u64);
    b.ext_mul_base_const(res, half_pow_l)
}

fn eval_eq_sharp_uni_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    omega_skip_pows: &[RootF],
    xi_1: &[BabyBearExtWire<B::F>],
    z: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let one = b.ext_from_base_const(RootF::ONE);
    let zero = b.ext_zero();
    let mut eq_xi_evals = vec![zero; 1usize << xi_1.len()];
    eq_xi_evals[0] = one;

    // Match `evals_eq_hypercube_serial` ordering from the native verifier:
    // mask bit `i` corresponds to `xi_1[i]`.
    for (i, xi) in xi_1.iter().enumerate() {
        let span = 1usize << i;
        let one_minus_xi = b.ext_sub(one, *xi);
        for idx in 0..span {
            let prev = eq_xi_evals[idx];
            let lo = b.ext_mul(prev, one_minus_xi);
            let hi = b.ext_mul(prev, *xi);
            eq_xi_evals[idx] = lo;
            eq_xi_evals[span + idx] = hi;
        }
    }

    assert_eq!(
        eq_xi_evals.len(),
        omega_skip_pows.len(),
        "eq_sharp eval table width mismatch",
    );

    let mut res = b.ext_zero();
    let l_skip = xi_1.len();
    for (omega_pow, eq_xi_eval) in omega_skip_pows.iter().zip(eq_xi_evals.iter()) {
        let omega_ext = b.ext_from_base_const(*omega_pow);
        let eq_uni = eval_eq_uni_assigned(b, l_skip, z, &omega_ext);
        let term = b.ext_mul(eq_uni, *eq_xi_eval);
        res = b.ext_add(res, term);
    }
    res
}

pub(crate) fn eval_eq_prism_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    l_skip: usize,
    x: &[BabyBearExtWire<B::F>],
    y: &[BabyBearExtWire<B::F>],
) -> BabyBearExtWire<B::F> {
    assert!(
        !x.is_empty() && !y.is_empty(),
        "eq_prism vectors must be non-empty",
    );
    let eq_uni = eval_eq_uni_assigned(b, l_skip, &x[0], &y[0]);
    let eq_mle = eval_eq_mle_assigned(b, &x[1..], &y[1..]);
    b.ext_mul(eq_uni, eq_mle)
}

fn eval_eq_rot_cube_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    x: &[BabyBearExtWire<B::F>],
    y: &[BabyBearExtWire<B::F>],
) -> (BabyBearExtWire<B::F>, BabyBearExtWire<B::F>) {
    assert_eq!(x.len(), y.len(), "eq_rot_cube vector length mismatch");
    let one = b.ext_from_base_const(RootF::ONE);
    let mut rot = one;
    let mut eq = one;
    for i in (0..x.len()).rev() {
        let one_minus_y = b.ext_sub(one, y[i]);
        let one_minus_x = b.ext_sub(one, x[i]);
        let x_times = b.ext_mul(x[i], one_minus_y);
        let term1 = b.ext_mul(x_times, eq);
        let y_times = b.ext_mul(one_minus_x, y[i]);
        let term2 = b.ext_mul(y_times, rot);
        rot = b.ext_add(term1, term2);

        let xy = b.ext_mul(x[i], y[i]);
        let one_minus_xy = b.ext_mul(one_minus_x, one_minus_y);
        let eq_factor = b.ext_add(xy, one_minus_xy);
        eq = b.ext_mul(eq, eq_factor);
    }
    (eq, rot)
}

pub(crate) fn eval_rot_kernel_prism_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    l_skip: usize,
    x: &[BabyBearExtWire<B::F>],
    y: &[BabyBearExtWire<B::F>],
) -> BabyBearExtWire<B::F> {
    assert!(
        !x.is_empty() && !y.is_empty(),
        "rot-kernel vectors must be non-empty",
    );
    let omega = RootF::two_adic_generator(l_skip);
    let y0_omega = b.ext_mul_base_const(y[0], omega);
    let eq_uni_rot = eval_eq_uni_assigned(b, l_skip, &x[0], &y0_omega);
    let (eq_cube, rot_cube) = eval_eq_rot_cube_assigned(b, &x[1..], &y[1..]);
    let term_a = b.ext_mul(eq_uni_rot, eq_cube);

    let eq_uni_x_one = eval_eq_uni_at_one_assigned(b, l_skip, &x[0]);
    let eq_uni_y_one = eval_eq_uni_at_one_assigned(b, l_skip, &y0_omega);
    let rot_minus_eq = b.ext_sub(rot_cube, eq_cube);
    let eq_uni_product = b.ext_mul(eq_uni_x_one, eq_uni_y_one);
    let term_b = b.ext_mul(eq_uni_product, rot_minus_eq);
    b.ext_add(term_a, term_b)
}

fn interpolate_linear_at_01_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    eval0: &BabyBearExtWire<B::F>,
    eval1: &BabyBearExtWire<B::F>,
    x: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let delta = b.ext_sub(*eval1, *eval0);
    let scaled = b.ext_mul(delta, *x);
    b.ext_add(scaled, *eval0)
}

fn interpolate_cubic_at_0123_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    evals: [&BabyBearExtWire<B::F>; 4],
    x: &BabyBearExtWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let inv6 = RootF::from_u64(6).inverse();
    let s1 = b.ext_sub(*evals[1], *evals[0]);
    let s2 = b.ext_sub(*evals[2], *evals[0]);
    let s3 = b.ext_sub(*evals[3], *evals[0]);

    let s2_minus_s1 = b.ext_sub(s2, s1);
    let triple = b.ext_mul_base_const(s2_minus_s1, RootF::from_u64(3));
    let d3 = b.ext_sub(s3, triple);

    let p = b.ext_mul_base_const(d3, inv6);
    let s2_minus_d3 = b.ext_sub(s2, d3);
    let half = RootF::ONE.halve();
    let q_half = b.ext_mul_base_const(s2_minus_d3, half);
    let q = b.ext_sub(q_half, s1);
    let p_plus_q = b.ext_add(p, q);
    let r = b.ext_sub(s1, p_plus_q);

    let p_mul_x = b.ext_mul(p, *x);
    let px_plus_q = b.ext_add(p_mul_x, q);
    let quad_mul_x = b.ext_mul(px_plus_q, *x);
    let quad = b.ext_add(quad_mul_x, r);
    let cubic = b.ext_mul(quad, *x);
    b.ext_add(cubic, *evals[0])
}

#[derive(Clone)]
struct ViewPairWire<F> {
    local: BabyBearExtWire<F>,
    next: BabyBearExtWire<F>,
}

impl<F> From<(BabyBearExtWire<F>, BabyBearExtWire<F>)> for ViewPairWire<F> {
    fn from((local, next): (BabyBearExtWire<F>, BabyBearExtWire<F>)) -> Self {
        Self { local, next }
    }
}

struct ConstraintEvaluatorWire<'a, F> {
    preprocessed: Option<&'a [ViewPairWire<F>]>,
    partitioned_main: &'a [Vec<ViewPairWire<F>>],
    is_first_row: BabyBearExtWire<F>,
    is_last_row: BabyBearExtWire<F>,
    public_values: &'a [ReducedBabyBearWire<F>],
}

impl<F: Copy> ConstraintEvaluatorWire<'_, F> {
    fn eval_var<B: BabyBearExt4Inst<F = F>>(
        &self,
        b: &mut B,
        symbolic_var: SymbolicVariable<RootF>,
    ) -> BabyBearExtWire<F> {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => {
                let value = &self.preprocessed.unwrap()[index];
                match offset {
                    0 => value.local,
                    1 => value.next,
                    _ => panic!("unsupported preprocessed rotation offset {offset}"),
                }
            }
            Entry::Main { part_index, offset } => {
                let value = &self.partitioned_main[part_index][index];
                match offset {
                    0 => value.local,
                    1 => value.next,
                    _ => panic!("unsupported main rotation offset {offset}"),
                }
            }
            Entry::Public => {
                let value = self.public_values[index];
                b.ext_from_base_var(value.into())
            }
            _ => panic!("invalid constraint"),
        }
    }
}

fn eval_symbolic_nodes_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    evaluator: &ConstraintEvaluatorWire<'_, B::F>,
    nodes: &[SymbolicExpressionNode<RootF>],
) -> Vec<BabyBearExtWire<B::F>> {
    let mut exprs: Vec<BabyBearExtWire<B::F>> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let expr = match node {
            SymbolicExpressionNode::Variable(var) => evaluator.eval_var(b, *var),
            SymbolicExpressionNode::Constant(c) => b.ext_from_base_const(*c),
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            } => b.ext_add(exprs[*left_idx], exprs[*right_idx]),
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => b.ext_sub(exprs[*left_idx], exprs[*right_idx]),
            SymbolicExpressionNode::Neg { idx, .. } => b.ext_neg(exprs[*idx]),
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => {
                let left_const = match &nodes[*left_idx] {
                    SymbolicExpressionNode::Constant(c) => Some(*c),
                    _ => None,
                };
                let right_const = match &nodes[*right_idx] {
                    SymbolicExpressionNode::Constant(c) => Some(*c),
                    _ => None,
                };
                match (left_const, right_const) {
                    (Some(lc), Some(rc)) => b.ext_from_base_const(lc * rc),
                    (Some(c), None) => b.ext_mul_base_const(exprs[*right_idx], c),
                    (None, Some(c)) => b.ext_mul_base_const(exprs[*left_idx], c),
                    (None, None) => b.ext_mul(exprs[*left_idx], exprs[*right_idx]),
                }
            }
            SymbolicExpressionNode::IsFirstRow => evaluator.is_first_row,
            SymbolicExpressionNode::IsLastRow => evaluator.is_last_row,
            SymbolicExpressionNode::IsTransition => {
                let one = b.ext_from_base_const(RootF::ONE);
                b.ext_sub(one, evaluator.is_last_row)
            }
        };
        exprs.push(expr);
    }
    exprs
}

fn local_next_opening_views<B: BabyBearExt4Inst>(
    b: &mut B,
    openings: &[ReducedBabyBearExtWire<B::F>],
    need_rot: bool,
) -> Vec<ViewPairWire<B::F>> {
    let openings = openings
        .iter()
        .map(BabyBearExtWire::from)
        .collect::<Vec<_>>();
    column_openings_by_rot_assigned(b, &openings, need_rot)
        .into_iter()
        .map(ViewPairWire::from)
        .collect()
}

fn observe_layer_claims_assigned<B: TranscriptInst>(
    b: &mut B,
    claims: &[ReducedBabyBearExtWire<B::F>],
) {
    for claim in claims {
        b.observe_ext(claim);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn constrain_batch_constraints_verification<B: TranscriptInst>(
    b: &mut B,
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    gkr_wire: &GkrProofWire<B::F>,
    batch_wire: &BatchConstraintProofWire<B::F>,
    n_per_trace: &[isize],
    trace_id_to_air_id: &[usize],
    public_values: Vec<Vec<ReducedBabyBearWire<B::F>>>,
    profiler: &mut CellProfiler,
) -> BatchConstraintIntermediatesWire<B::F> {
    let l_skip = mvk0.params.l_skip;

    let trace_id_to_air_id_host = trace_id_to_air_id.to_vec();
    let total_interactions_host = zip(trace_id_to_air_id, n_per_trace)
        .map(|(&air_idx, &n)| {
            let n_lift = n.max(0) as usize;
            let num_interactions = mvk0.per_air[air_idx]
                .symbolic_constraints
                .interactions
                .len();
            (num_interactions as u64) << (l_skip + n_lift)
        })
        .sum::<u64>();
    assert!(total_interactions_host > 0, "0 interactions not supported");
    let n_logup_host = calculate_n_logup(l_skip, total_interactions_host);
    let n_max_host = n_per_trace.iter().copied().max().unwrap().max(0) as usize;
    let n_global_host = n_max_host.max(n_logup_host);
    let omega_skip = RootF::two_adic_generator(l_skip);
    let omega_skip_pows: Vec<_> = omega_skip.powers().take(1usize << l_skip).collect();

    let trace_has_preprocessed = trace_id_to_air_id
        .iter()
        .map(|&air_id| mvk0.per_air[air_id].preprocessed_data.is_some())
        .collect::<Vec<_>>();
    let trace_constraint_nodes = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            mvk0.per_air[air_id]
                .symbolic_constraints
                .constraints
                .nodes
                .clone()
        })
        .collect::<Vec<_>>();
    let trace_constraint_indices = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            mvk0.per_air[air_id]
                .symbolic_constraints
                .constraints
                .constraint_idx
                .clone()
        })
        .collect::<Vec<_>>();
    let trace_interactions = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            mvk0.per_air[air_id]
                .symbolic_constraints
                .interactions
                .clone()
        })
        .collect::<Vec<_>>();
    let column_openings_need_rot = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            let need_rot = mvk0.per_air[air_id].params.need_rot;
            vec![need_rot; mvk0.per_air[air_id].num_parts()]
        })
        .collect::<Vec<_>>();

    let logup_pow_bits = mvk0.params.logup.pow_bits;
    let logup_pow_witness = gkr_wire.logup_pow_witness;
    b.check_witness(logup_pow_bits, &logup_pow_witness);

    let alpha_logup = b.sample_ext();
    let beta_logup = b.sample_ext();

    profiler.push("gkr_verification", b.cell_count());

    let gkr_claims_per_layer = &gkr_wire.claims_per_layer;
    let gkr_sumcheck_polys = &gkr_wire.sumcheck_polys;

    let one = b.ext_from_base_const(RootF::ONE);
    let total_gkr_rounds = l_skip + n_logup_host;
    let (mut gkr_p_xi_claim, mut gkr_q_xi_claim, mut xi) = {
        let gkr_q0_claim = gkr_wire.q0_claim;
        b.observe_ext(&gkr_q0_claim);

        let layer0 = &gkr_claims_per_layer[0];
        observe_layer_claims_assigned(b, layer0);

        let layer0_p0 = layer0[0].into();
        let layer0_q0 = layer0[1].into();
        let layer0_p1 = layer0[2].into();
        let layer0_q1 = layer0[3].into();
        let p0_q1 = b.ext_mul(layer0_p0, layer0_q1);
        let p1_q0 = b.ext_mul(layer0_p1, layer0_q0);
        let p_cross = b.ext_add(p0_q1, p1_q0);
        let q_cross = b.ext_mul(layer0_q0, layer0_q1);
        b.ext_assert_zero(p_cross);
        b.ext_assert_equal(q_cross, gkr_q0_claim.into());

        let mu0 = b.sample_ext();
        let mut numer_claim = interpolate_linear_at_01_assigned(b, &layer0_p0, &layer0_p1, &mu0);
        let mut denom_claim = interpolate_linear_at_01_assigned(b, &layer0_q0, &layer0_q1, &mu0);
        let mut gkr_r = vec![mu0];

        for round in 1..total_gkr_rounds {
            let lambda_round = b.sample_ext();

            let lambda_denom = b.ext_mul(lambda_round, denom_claim);
            let mut claim = b.ext_add(numer_claim, lambda_denom);
            let round_polys = &gkr_sumcheck_polys[round - 1];
            let mut gkr_r_prime = Vec::with_capacity(round);
            let mut eq = one;

            for (subround, xi_prev) in gkr_r.iter().enumerate().take(round) {
                let [ev1, ev2, ev3] = round_polys[subround];
                b.observe_ext(&ev1);
                b.observe_ext(&ev2);
                b.observe_ext(&ev3);

                let ri = b.sample_ext();
                gkr_r_prime.push(ri);

                let ev1 = ev1.into();
                let ev2 = ev2.into();
                let ev3 = ev3.into();
                let ev0 = b.ext_sub(claim, ev1);
                claim = interpolate_cubic_at_0123_assigned(b, [&ev0, &ev1, &ev2, &ev3], &ri);
                let xi_ri = b.ext_mul(*xi_prev, ri);
                let one_minus_xi = b.ext_sub(one, *xi_prev);
                let one_minus_ri = b.ext_sub(one, ri);
                let one_minus_term = b.ext_mul(one_minus_xi, one_minus_ri);
                let eq_factor = b.ext_add(xi_ri, one_minus_term);
                eq = b.ext_mul(eq, eq_factor);
            }

            let layer_claims = &gkr_claims_per_layer[round];
            observe_layer_claims_assigned(b, layer_claims);

            let layer_p0 = layer_claims[0].into();
            let layer_q0 = layer_claims[1].into();
            let layer_p1 = layer_claims[2].into();
            let layer_q1 = layer_claims[3].into();
            let p0_q1 = b.ext_mul(layer_p0, layer_q1);
            let p1_q0 = b.ext_mul(layer_p1, layer_q0);
            let p_cross = b.ext_add(p0_q1, p1_q0);
            let q_cross = b.ext_mul(layer_q0, layer_q1);
            let lambda_q_cross = b.ext_mul(lambda_round, q_cross);
            let claim_sum = b.ext_add(p_cross, lambda_q_cross);
            let expected_claim = b.ext_mul(claim_sum, eq);
            b.ext_assert_equal(expected_claim, claim);

            let mu_round = b.sample_ext();
            numer_claim = interpolate_linear_at_01_assigned(b, &layer_p0, &layer_p1, &mu_round);
            denom_claim = interpolate_linear_at_01_assigned(b, &layer_q0, &layer_q1, &mu_round);
            gkr_r = core::iter::once(mu_round)
                .chain(gkr_r_prime.into_iter())
                .collect();
        }

        (numer_claim, denom_claim, gkr_r)
    };

    while xi.len() != l_skip + n_global_host {
        xi.push(b.sample_ext());
    }

    let lambda = b.sample_ext();

    profiler.pop(b.cell_count());
    profiler.push("batch_sumcheck", b.cell_count());

    let numerator_term_per_air = &batch_wire.numerator_term_per_air;
    let denominator_term_per_air = &batch_wire.denominator_term_per_air;
    for (num_term, den_term) in numerator_term_per_air
        .iter()
        .zip(denominator_term_per_air.iter())
    {
        gkr_p_xi_claim = b.ext_sub(gkr_p_xi_claim, num_term.into());
        gkr_q_xi_claim = b.ext_sub(gkr_q_xi_claim, den_term.into());
        b.observe_ext(num_term);
        b.observe_ext(den_term);
    }
    let gkr_numerator_residual = gkr_p_xi_claim;
    let gkr_denominator_claim = gkr_q_xi_claim;
    b.ext_assert_zero(gkr_numerator_residual);
    b.ext_assert_equal(gkr_denominator_claim, alpha_logup);

    let mu = b.sample_ext();

    let mut sum_claim = b.ext_zero();
    let mut cur_mu_pow = one;
    let mut first_mu_term = true;
    for (num_term, den_term) in numerator_term_per_air
        .iter()
        .zip(denominator_term_per_air.iter())
    {
        let num_term = num_term.into();
        let den_term = den_term.into();
        let num_weighted = if first_mu_term {
            first_mu_term = false;
            num_term
        } else {
            b.ext_mul(num_term, cur_mu_pow)
        };
        sum_claim = b.ext_add(sum_claim, num_weighted);
        cur_mu_pow = b.ext_mul(cur_mu_pow, mu);

        let den_weighted = b.ext_mul(den_term, cur_mu_pow);
        sum_claim = b.ext_add(sum_claim, den_weighted);
        cur_mu_pow = b.ext_mul(cur_mu_pow, mu);
    }

    let univariate_round_coeffs = &batch_wire.univariate_round_coeffs;
    for coeff in univariate_round_coeffs {
        b.observe_ext(coeff);
    }
    let univariate_round_coeffs_raw = univariate_round_coeffs
        .iter()
        .map(BabyBearExtWire::from)
        .collect::<Vec<_>>();
    let mut r = vec![b.sample_ext()];

    let stride = 1usize << l_skip;
    let mut sum_univ_domain_s_0 = b.ext_zero();
    for coeff in univariate_round_coeffs_raw.iter().step_by(stride) {
        sum_univ_domain_s_0 = b.ext_add(sum_univ_domain_s_0, *coeff);
    }
    let sum_univ_domain_s_0 =
        b.ext_mul_base_const(sum_univ_domain_s_0, RootF::from_u64(stride as u64));
    b.ext_assert_equal(sum_claim, sum_univ_domain_s_0);

    let sumcheck_round_polys = &batch_wire.sumcheck_round_polys;
    let mut consistency_lhs = horner_eval_ext_poly_assigned(b, &univariate_round_coeffs_raw, &r[0]);
    for round_evals in sumcheck_round_polys {
        for eval in round_evals {
            b.observe_ext(eval);
        }

        let s_1 = round_evals[0].into();
        let s_0 = b.ext_sub(consistency_lhs, s_1);
        let mut interpolation_evals = Vec::with_capacity(round_evals.len() + 1);
        interpolation_evals.push(s_0);
        interpolation_evals.extend(round_evals.iter().map(BabyBearExtWire::from));
        let next_r = b.sample_ext();
        consistency_lhs = eval_lagrange_on_integer_grid(b, &next_r, &interpolation_evals);
        r.push(next_r);
    }

    profiler.pop(b.cell_count());
    profiler.push("observe_openings", b.cell_count());

    let column_openings = &batch_wire.column_openings;

    let reduced_zero = b.ext_load_reduced_constant(RootEF::ZERO);
    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let need_rot = column_openings_need_rot[trace_idx][0];
        let openings = &air_openings[0];
        if need_rot {
            assert!(
                openings.len().is_multiple_of(2),
                "rotated opening vector must be even",
            );
            for claim in openings.chunks_exact(2) {
                b.observe_ext(&claim[0]);
                b.observe_ext(&claim[1]);
            }
        } else {
            for opening in openings {
                b.observe_ext(opening);
                b.observe_ext(&reduced_zero);
            }
        }
    }

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        for (part_idx, claims) in air_openings.iter().enumerate().skip(1) {
            let need_rot = column_openings_need_rot[trace_idx][part_idx];
            if need_rot {
                assert!(
                    claims.len().is_multiple_of(2),
                    "rotated opening vector must be even",
                );
                for claim in claims.chunks_exact(2) {
                    b.observe_ext(&claim[0]);
                    b.observe_ext(&claim[1]);
                }
            } else {
                for claim in claims {
                    b.observe_ext(claim);
                    b.observe_ext(&reduced_zero);
                }
            }
        }
    }

    profiler.pop(b.cell_count());
    profiler.push("eq_3b_tree", b.cell_count());

    let mut eq_3b_per_trace = Vec::with_capacity(n_per_trace.len());
    let mut stacked_idx = 0usize;
    for (trace_idx, &n) in n_per_trace.iter().enumerate() {
        let n_lift = n.max(0) as usize;
        let interactions = &trace_interactions[trace_idx];
        if interactions.is_empty() {
            eq_3b_per_trace.push(Vec::new());
            continue;
        }

        let d = n_logup_host.saturating_sub(n_lift);
        let xi_slice = &xi[l_skip + n_lift..l_skip + n_logup_host];

        // Determine needed leaf indices before building the tree.
        let needed_leaves: Vec<usize> = {
            let mut leaves = Vec::with_capacity(interactions.len());
            let mut tmp_idx = stacked_idx;
            for _ in 0..interactions.len() {
                let b_int = tmp_idx >> (l_skip + n_lift);
                let tree_idx = b_int & ((1 << d) - 1);
                leaves.push(tree_idx);
                tmp_idx += 1 << (l_skip + n_lift);
            }
            leaves
        };

        // Precompute per-bit factors: (x_i, 1-x_i) for tree product.
        let factors: Vec<_> = xi_slice
            .iter()
            .map(|x_i| {
                let one_minus_x = b.ext_sub(one, *x_i);
                (*x_i, one_minus_x)
            })
            .collect();

        // Build a sparse partial product tree containing only ancestors of needed leaves.
        // tree[level][node_idx] = product of factors for bits matching node_idx.
        // Level 0: single root node with value `one`.
        // Level j+1: only children whose index appears in the needed set for that level.
        let mut prev_level: BTreeMap<usize, BabyBearExtWire<B::F>> = BTreeMap::new();
        prev_level.insert(0, one);

        for level_idx in 0..d {
            let factor_j = d - 1 - level_idx;
            // Determine which nodes are needed at level (level_idx + 1):
            // a leaf index shifted right by the remaining levels.
            let shift = d - (level_idx + 1);
            let mut curr_level = BTreeMap::new();
            for node_idx in needed_leaves.iter().map(|&leaf| leaf >> shift) {
                if curr_level.contains_key(&node_idx) {
                    continue;
                }
                let parent_idx = node_idx >> 1;
                let parent = prev_level[&parent_idx];
                let val = if node_idx & 1 == 0 {
                    b.ext_mul(parent, factors[factor_j].1)
                } else {
                    b.ext_mul(parent, factors[factor_j].0)
                };
                curr_level.insert(node_idx, val);
            }
            prev_level = curr_level;
        }

        // Look up each interaction's eq_3b from the sparse tree leaves.
        let mut eq_3b = Vec::with_capacity(interactions.len());
        for &tree_idx in &needed_leaves {
            stacked_idx += 1 << (l_skip + n_lift);
            eq_3b.push(prev_level[&tree_idx]);
        }
        eq_3b_per_trace.push(eq_3b);
    }

    profiler.pop(b.cell_count());
    profiler.push("eq_ns_precompute", b.cell_count());

    let mut eq_ns = vec![one; n_max_host + 1];
    let mut eq_sharp_ns = vec![one; n_max_host + 1];
    eq_ns[0] = eval_eq_uni_assigned(b, l_skip, &xi[0], &r[0]);
    eq_sharp_ns[0] = eval_eq_sharp_uni_assigned(b, &omega_skip_pows, &xi[..l_skip], &r[0]);
    for (i, r_i) in r.iter().enumerate().skip(1) {
        let eq_mle = eval_eq_mle_assigned(b, &[xi[l_skip + i - 1]], core::slice::from_ref(r_i));
        eq_ns[i] = b.ext_mul(eq_ns[i - 1], eq_mle);
        eq_sharp_ns[i] = b.ext_mul(eq_sharp_ns[i - 1], eq_mle);
        eq_ns[i] = b.ext_reduce_max_bits(eq_ns[i]);
        eq_sharp_ns[i] = b.ext_reduce_max_bits(eq_sharp_ns[i]);
    }
    if n_max_host > 0 {
        let n_max_usize = n_max_host;
        let mut r_rev_prod = r[n_max_usize];
        for i in (0..n_max_usize).rev() {
            eq_ns[i] = b.ext_mul(eq_ns[i], r_rev_prod);
            eq_sharp_ns[i] = b.ext_mul(eq_sharp_ns[i], r_rev_prod);
            eq_ns[i] = b.ext_reduce_max_bits(eq_ns[i]);
            eq_sharp_ns[i] = b.ext_reduce_max_bits(eq_sharp_ns[i]);
            r_rev_prod = b.ext_mul(r_rev_prod, r[i]);
        }
    }

    profiler.pop(b.cell_count());
    profiler.push("constraint_eval", b.cell_count());

    let mut interactions_evals = Vec::new();
    let mut constraints_evals = Vec::new();

    let mut beta_pows = vec![one];
    let mut lambda_pows = vec![one];
    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let air_idx = trace_id_to_air_id_host[trace_idx];
        let n = n_per_trace[trace_idx];
        let n_lift = n.max(0) as usize;

        let need_rot_flags = &column_openings_need_rot[trace_idx];
        let common_main = local_next_opening_views(b, &air_openings[0], need_rot_flags[0]);
        let has_preprocessed = trace_has_preprocessed[trace_idx];
        let preprocessed = has_preprocessed
            .then(|| local_next_opening_views(b, &air_openings[1], need_rot_flags[1]));
        let cached_idx = 1 + has_preprocessed as usize;
        let mut partitioned_main = air_openings[cached_idx..]
            .iter()
            .enumerate()
            .map(|(part_offset, opening)| {
                local_next_opening_views(b, opening, need_rot_flags[cached_idx + part_offset])
            })
            .collect::<Vec<_>>();
        partitioned_main.push(common_main);

        let (l, rs_n, norm_factor) = if n.is_negative() {
            (
                l_skip.wrapping_add_signed(n),
                vec![b.ext_pow_power_of_two(r[0], n.unsigned_abs())],
                RootF::from_usize(1usize << n.unsigned_abs()).inverse(),
            )
        } else {
            (l_skip, r[..=n_lift].to_vec(), RootF::ONE)
        };

        let inv_l = RootF::from_usize(1usize << l).inverse();
        let mut is_first_row = progression_exp_2_assigned(b, &rs_n[0], l);
        is_first_row = b.ext_mul_base_const(is_first_row, inv_l);
        for x in rs_n.iter().skip(1) {
            let one_minus_x = b.ext_sub(one, *x);
            is_first_row = b.ext_mul(is_first_row, one_minus_x);
        }

        let omega = RootF::two_adic_generator(l);
        let rs0_omega = b.ext_mul_base_const(rs_n[0], omega);
        let mut is_last_row = progression_exp_2_assigned(b, &rs0_omega, l);
        is_last_row = b.ext_mul_base_const(is_last_row, inv_l);
        for x in rs_n.iter().skip(1) {
            is_last_row = b.ext_mul(is_last_row, *x);
        }

        let is_first_row = b.ext_reduce_max_bits(is_first_row);
        let is_last_row = b.ext_reduce_max_bits(is_last_row);
        let evaluator = ConstraintEvaluatorWire {
            preprocessed: preprocessed.as_deref(),
            partitioned_main: &partitioned_main,
            is_first_row,
            is_last_row,
            public_values: public_values[air_idx].as_slice(),
        };

        let node_values =
            eval_symbolic_nodes_assigned(b, &evaluator, &trace_constraint_nodes[trace_idx]);

        let mut expr = b.ext_zero();
        for (i, &constraint_idx) in trace_constraint_indices[trace_idx].iter().enumerate() {
            let term = if i == 0 {
                node_values[constraint_idx]
            } else {
                if i >= lambda_pows.len() {
                    debug_assert_eq!(i, lambda_pows.len());
                    let new_pow = b.ext_mul(*lambda_pows.last().unwrap(), lambda);
                    let new_pow = b.ext_reduce_max_bits(new_pow);
                    lambda_pows.push(new_pow);
                }

                b.ext_mul(node_values[constraint_idx], lambda_pows[i])
            };
            expr = b.ext_add(expr, term);
        }
        let constraints_eval = b.ext_mul(eq_ns[n_lift], expr);
        constraints_evals.push(constraints_eval);

        let interactions = &trace_interactions[trace_idx];
        let eq_3bs = &eq_3b_per_trace[trace_idx];
        let mut num = b.ext_zero();
        let mut denom = b.ext_zero();
        for (eq_3b, interaction) in eq_3bs.iter().zip(interactions.iter()) {
            let count_eval = node_values[interaction.count];
            let mut denom_eval = b.ext_zero();
            for (j, &msg_idx) in interaction.message.iter().enumerate() {
                let term = if j == 0 {
                    node_values[msg_idx]
                } else {
                    if j >= beta_pows.len() {
                        debug_assert_eq!(j, beta_pows.len());
                        let new_pow = b.ext_mul(*beta_pows.last().unwrap(), beta_logup);
                        let new_pow = b.ext_reduce_max_bits(new_pow);
                        beta_pows.push(new_pow);
                    }

                    b.ext_mul(node_values[msg_idx], beta_pows[j])
                };
                denom_eval = b.ext_add(denom_eval, term);
            }
            if interaction.message.len() >= beta_pows.len() {
                let new_pow = b.ext_mul(*beta_pows.last().unwrap(), beta_logup);
                let new_pow = b.ext_reduce_max_bits(new_pow);
                beta_pows.push(new_pow);
            }
            let bus_term = b.ext_mul_base_const(
                beta_pows[interaction.message.len()],
                RootF::from_u64(u64::from(interaction.bus_index) + 1),
            );
            denom_eval = b.ext_add(denom_eval, bus_term);

            let eq_times_count = b.ext_mul(*eq_3b, count_eval);
            num = b.ext_add(num, eq_times_count);
            let eq_times_denom = b.ext_mul(*eq_3b, denom_eval);
            denom = b.ext_add(denom, eq_times_denom);
        }

        let num_norm = if norm_factor == RootF::ONE {
            num
        } else {
            b.ext_mul_base_const(num, norm_factor)
        };
        let num_scaled = b.ext_mul(num_norm, eq_sharp_ns[n_lift]);
        let denom_scaled = b.ext_mul(denom, eq_sharp_ns[n_lift]);
        interactions_evals.push(num_scaled);
        interactions_evals.push(denom_scaled);
    }

    profiler.pop(b.cell_count());
    profiler.push("final_consistency", b.cell_count());

    let mut consistency_rhs = b.ext_zero();
    let mut cur_mu_pow = one;
    for (i, term) in interactions_evals
        .iter()
        .chain(constraints_evals.iter())
        .enumerate()
    {
        let weighted_term = if i == 0 {
            *term
        } else {
            b.ext_mul(*term, cur_mu_pow)
        };
        consistency_rhs = b.ext_add(consistency_rhs, weighted_term);
        cur_mu_pow = b.ext_mul(cur_mu_pow, mu);
    }
    b.ext_assert_equal(consistency_lhs, consistency_rhs);

    profiler.pop(b.cell_count());

    BatchConstraintIntermediatesWire {
        column_openings: column_openings.clone(),
        r,
    }
}
