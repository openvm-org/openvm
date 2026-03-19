use std::{iter::zip, sync::Arc};

use halo2_base::Context;
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
    field::baby_bear::{BabyBearChip, BabyBearExtChip, BabyBearExtWire, BabyBearWire},
    stages::shared_math,
    transcript::TranscriptGadget,
    Fr, RootF,
};

#[derive(Clone, Debug)]
pub struct BatchConstraintWire {
    pub column_openings: Vec<Vec<Vec<BabyBearExtWire>>>,
    pub r: Vec<BabyBearExtWire>,
}

#[derive(Clone, Debug)]
pub struct GkrProofWire {
    pub logup_pow_witness: BabyBearWire,
    pub q0_claim: BabyBearExtWire,
    pub claims_per_layer: Vec<Vec<BabyBearExtWire>>,
    pub sumcheck_polys: Vec<Vec<BabyBearExtWire>>,
}

#[derive(Clone, Debug)]
pub struct BatchConstraintProofWire {
    pub numerator_term_per_air: Vec<BabyBearExtWire>,
    pub denominator_term_per_air: Vec<BabyBearExtWire>,
    pub univariate_round_coeffs: Vec<BabyBearExtWire>,
    pub sumcheck_round_polys: Vec<Vec<BabyBearExtWire>>,
    pub column_openings: Vec<Vec<Vec<BabyBearExtWire>>>,
}

pub(crate) fn load_gkr_proof_wire(
    ctx: &mut Context<Fr>,
    base_chip: &BabyBearChip,
    ext_chip: &BabyBearExtChip,
    gkr_proof: &openvm_stark_sdk::openvm_stark_backend::proof::GkrProof<RootConfig>,
) -> GkrProofWire {
    let logup_pow_witness = base_chip.load_witness(ctx, gkr_proof.logup_pow_witness);
    let q0_claim = ext_chip.load_witness(ctx, gkr_proof.q0_claim);
    let claims_per_layer = gkr_proof
        .claims_per_layer
        .iter()
        .map(|claims| {
            vec![
                ext_chip.load_witness(ctx, claims.p_xi_0),
                ext_chip.load_witness(ctx, claims.q_xi_0),
                ext_chip.load_witness(ctx, claims.p_xi_1),
                ext_chip.load_witness(ctx, claims.q_xi_1),
            ]
        })
        .collect::<Vec<_>>();
    let sumcheck_polys = gkr_proof
        .sumcheck_polys
        .iter()
        .map(|poly| {
            let mut assigned = Vec::with_capacity(poly.len() * 3);
            for evals in poly {
                for &value in evals {
                    assigned.push(ext_chip.load_witness(ctx, value));
                }
            }
            assigned
        })
        .collect::<Vec<_>>();
    GkrProofWire {
        logup_pow_witness,
        q0_claim,
        claims_per_layer,
        sumcheck_polys,
    }
}

pub(crate) fn load_batch_constraint_proof_wire(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    batch_proof: &openvm_stark_sdk::openvm_stark_backend::proof::BatchConstraintProof<RootConfig>,
) -> BatchConstraintProofWire {
    let numerator_term_per_air = batch_proof
        .numerator_term_per_air
        .iter()
        .map(|&value| ext_chip.load_witness(ctx, value))
        .collect::<Vec<_>>();
    let denominator_term_per_air = batch_proof
        .denominator_term_per_air
        .iter()
        .map(|&value| ext_chip.load_witness(ctx, value))
        .collect::<Vec<_>>();
    let univariate_round_coeffs = batch_proof
        .univariate_round_coeffs
        .iter()
        .map(|&value| ext_chip.load_witness(ctx, value))
        .collect::<Vec<_>>();
    let sumcheck_round_polys = batch_proof
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| ext_chip.load_witness(ctx, value))
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
                        .map(|&value| ext_chip.load_witness(ctx, value))
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

fn eval_ext_poly_horner(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    coeffs: &[BabyBearExtWire],
    point: &BabyBearExtWire,
) -> BabyBearExtWire {
    let mut acc = ext_chip.zero(ctx);
    for coeff in coeffs.iter().rev() {
        acc = ext_chip.mul(ctx, acc, *point);
        acc = ext_chip.add(ctx, acc, *coeff);
    }
    acc
}

fn eval_lagrange_on_integer_grid(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    point: &BabyBearExtWire,
    evals: &[BabyBearExtWire],
) -> BabyBearExtWire {
    let n = evals.len().saturating_sub(1);
    let mut acc = ext_chip.zero(ctx);
    for (i, eval_i) in evals.iter().enumerate() {
        let mut basis = ext_chip.from_base_const(ctx, RootF::ONE);
        let mut denom = RootF::ONE;
        for j in 0..=n {
            if i == j {
                continue;
            }
            let x_j = ext_chip.from_base_const(ctx, RootF::from_u64(j as u64));
            let x_minus_j = ext_chip.sub(ctx, *point, x_j);
            basis = ext_chip.mul(ctx, basis, x_minus_j);

            let diff = if i >= j {
                RootF::from_usize(i - j)
            } else {
                -RootF::from_usize(j - i)
            };
            denom *= diff;
        }
        let denom_inv = denom.inverse();
        let basis = ext_chip.mul_base_const(ctx, basis, denom_inv);
        let term = ext_chip.mul(ctx, *eval_i, basis);
        acc = ext_chip.add(ctx, acc, term);
    }
    acc
}

fn progression_exp_2_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    m: &BabyBearExtWire,
    l: usize,
) -> BabyBearExtWire {
    let mut pow = *m;
    let mut sum = ext_chip.from_base_const(ctx, RootF::ONE);
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    for _ in 0..l {
        let one_plus_pow = ext_chip.add(ctx, one, pow);
        sum = ext_chip.mul(ctx, sum, one_plus_pow);
        pow = ext_chip.mul(ctx, pow, pow);
    }
    sum
}

pub(crate) fn eval_eq_mle_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    x: &[BabyBearExtWire],
    y: &[BabyBearExtWire],
) -> BabyBearExtWire {
    assert_eq!(x.len(), y.len(), "eq_mle vector length mismatch");
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut acc = one;
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        let xy = ext_chip.mul(ctx, *x_i, *y_i);
        let two_xy = ext_chip.mul_base_const(ctx, xy, RootF::TWO);
        let one_minus_y = ext_chip.sub(ctx, one, *y_i);
        let one_minus_y_minus_x = ext_chip.sub(ctx, one_minus_y, *x_i);
        let factor = ext_chip.add(ctx, one_minus_y_minus_x, two_xy);
        acc = ext_chip.mul(ctx, acc, factor);
    }
    acc
}

pub(crate) fn eval_eq_mle_binary_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    x: &[BabyBearExtWire],
    y_bits: &[bool],
) -> BabyBearExtWire {
    assert_eq!(
        x.len(),
        y_bits.len(),
        "eq_mle binary vector length mismatch",
    );
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut acc = one;
    for (x_i, bit) in x.iter().zip(y_bits.iter().copied()) {
        let factor = if bit {
            *x_i
        } else {
            ext_chip.sub(ctx, one, *x_i)
        };
        acc = ext_chip.mul(ctx, acc, factor);
    }
    acc
}

pub(crate) fn eval_eq_uni_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    l_skip: usize,
    x: &BabyBearExtWire,
    y: &BabyBearExtWire,
) -> BabyBearExtWire {
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut res = one;
    let mut x_pow = *x;
    let mut y_pow = *y;
    for _ in 0..l_skip {
        let x_plus_y = ext_chip.add(ctx, x_pow, y_pow);
        let x_minus_one = ext_chip.sub(ctx, x_pow, one);
        let y_minus_one = ext_chip.sub(ctx, y_pow, one);
        let correction = ext_chip.mul(ctx, x_minus_one, y_minus_one);
        let scaled_res = ext_chip.mul(ctx, x_plus_y, res);
        res = ext_chip.add(ctx, scaled_res, correction);
        x_pow = ext_chip.mul(ctx, x_pow, x_pow);
        y_pow = ext_chip.mul(ctx, y_pow, y_pow);
    }
    let half_pow_l = RootF::ONE.halve().exp_u64(l_skip as u64);
    ext_chip.mul_base_const(ctx, res, half_pow_l)
}

pub(crate) fn eval_eq_uni_at_one_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    l_skip: usize,
    x: &BabyBearExtWire,
) -> BabyBearExtWire {
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut res = one;
    let mut x_pow = *x;
    for _ in 0..l_skip {
        let x_plus_one = ext_chip.add(ctx, x_pow, one);
        res = ext_chip.mul(ctx, res, x_plus_one);
        x_pow = ext_chip.mul(ctx, x_pow, x_pow);
    }
    let half_pow_l = RootF::ONE.halve().exp_u64(l_skip as u64);
    ext_chip.mul_base_const(ctx, res, half_pow_l)
}

fn eval_eq_sharp_uni_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    omega_skip_pows: &[RootF],
    xi_1: &[BabyBearExtWire],
    z: &BabyBearExtWire,
) -> BabyBearExtWire {
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut eq_xi_evals = vec![ext_chip.zero(ctx); 1usize << xi_1.len()];
    eq_xi_evals[0] = one;

    // Match `evals_eq_hypercube_serial` ordering from the native verifier:
    // mask bit `i` corresponds to `xi_1[i]`.
    for (i, xi) in xi_1.iter().enumerate() {
        let span = 1usize << i;
        let one_minus_xi = ext_chip.sub(ctx, one, *xi);
        for idx in 0..span {
            let prev = eq_xi_evals[idx];
            let lo = ext_chip.mul(ctx, prev, one_minus_xi);
            let hi = ext_chip.mul(ctx, prev, *xi);
            eq_xi_evals[idx] = lo;
            eq_xi_evals[span + idx] = hi;
        }
    }

    assert_eq!(
        eq_xi_evals.len(),
        omega_skip_pows.len(),
        "eq_sharp eval table width mismatch",
    );

    let mut res = ext_chip.zero(ctx);
    let l_skip = xi_1.len();
    for (omega_pow, eq_xi_eval) in omega_skip_pows.iter().zip(eq_xi_evals.iter()) {
        let omega_ext = ext_chip.from_base_const(ctx, *omega_pow);
        let eq_uni = eval_eq_uni_assigned(ctx, ext_chip, l_skip, z, &omega_ext);
        let term = ext_chip.mul(ctx, eq_uni, *eq_xi_eval);
        res = ext_chip.add(ctx, res, term);
    }
    res
}

pub(crate) fn eval_eq_prism_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    l_skip: usize,
    x: &[BabyBearExtWire],
    y: &[BabyBearExtWire],
) -> BabyBearExtWire {
    assert!(
        !x.is_empty() && !y.is_empty(),
        "eq_prism vectors must be non-empty",
    );
    let eq_uni = eval_eq_uni_assigned(ctx, ext_chip, l_skip, &x[0], &y[0]);
    let eq_mle = eval_eq_mle_assigned(ctx, ext_chip, &x[1..], &y[1..]);
    ext_chip.mul(ctx, eq_uni, eq_mle)
}

fn eval_eq_rot_cube_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    x: &[BabyBearExtWire],
    y: &[BabyBearExtWire],
) -> (BabyBearExtWire, BabyBearExtWire) {
    assert_eq!(x.len(), y.len(), "eq_rot_cube vector length mismatch");
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut rot = one;
    let mut eq = one;
    for i in (0..x.len()).rev() {
        let one_minus_y = ext_chip.sub(ctx, one, y[i]);
        let one_minus_x = ext_chip.sub(ctx, one, x[i]);
        let x_times = ext_chip.mul(ctx, x[i], one_minus_y);
        let term1 = ext_chip.mul(ctx, x_times, eq);
        let y_times = ext_chip.mul(ctx, one_minus_x, y[i]);
        let term2 = ext_chip.mul(ctx, y_times, rot);
        rot = ext_chip.add(ctx, term1, term2);

        let xy = ext_chip.mul(ctx, x[i], y[i]);
        let one_minus_xy = ext_chip.mul(ctx, one_minus_x, one_minus_y);
        let eq_factor = ext_chip.add(ctx, xy, one_minus_xy);
        eq = ext_chip.mul(ctx, eq, eq_factor);
    }
    (eq, rot)
}

pub(crate) fn eval_rot_kernel_prism_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    l_skip: usize,
    x: &[BabyBearExtWire],
    y: &[BabyBearExtWire],
) -> BabyBearExtWire {
    assert!(
        !x.is_empty() && !y.is_empty(),
        "rot-kernel vectors must be non-empty",
    );
    let omega = RootF::two_adic_generator(l_skip);
    let y0_omega = ext_chip.mul_base_const(ctx, y[0], omega);
    let eq_uni_rot = eval_eq_uni_assigned(ctx, ext_chip, l_skip, &x[0], &y0_omega);
    let (eq_cube, rot_cube) = eval_eq_rot_cube_assigned(ctx, ext_chip, &x[1..], &y[1..]);
    let term_a = ext_chip.mul(ctx, eq_uni_rot, eq_cube);

    let eq_uni_x_one = eval_eq_uni_at_one_assigned(ctx, ext_chip, l_skip, &x[0]);
    let eq_uni_y_one = eval_eq_uni_at_one_assigned(ctx, ext_chip, l_skip, &y0_omega);
    let rot_minus_eq = ext_chip.sub(ctx, rot_cube, eq_cube);
    let eq_uni_product = ext_chip.mul(ctx, eq_uni_x_one, eq_uni_y_one);
    let term_b = ext_chip.mul(ctx, eq_uni_product, rot_minus_eq);
    ext_chip.add(ctx, term_a, term_b)
}

fn interpolate_linear_at_01_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    eval0: &BabyBearExtWire,
    eval1: &BabyBearExtWire,
    x: &BabyBearExtWire,
) -> BabyBearExtWire {
    let delta = ext_chip.sub(ctx, *eval1, *eval0);
    let scaled = ext_chip.mul(ctx, delta, *x);
    ext_chip.add(ctx, scaled, *eval0)
}

fn interpolate_cubic_at_0123_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    evals: [&BabyBearExtWire; 4],
    x: &BabyBearExtWire,
) -> BabyBearExtWire {
    let inv6 = RootF::from_u64(6).inverse();
    let s1 = ext_chip.sub(ctx, *evals[1], *evals[0]);
    let s2 = ext_chip.sub(ctx, *evals[2], *evals[0]);
    let s3 = ext_chip.sub(ctx, *evals[3], *evals[0]);

    let s2_minus_s1 = ext_chip.sub(ctx, s2, s1);
    let triple = ext_chip.mul_base_const(ctx, s2_minus_s1, RootF::from_u64(3));
    let d3 = ext_chip.sub(ctx, s3, triple);

    let p = ext_chip.mul_base_const(ctx, d3, inv6);
    let s2_minus_d3 = ext_chip.sub(ctx, s2, d3);
    let half = RootF::ONE.halve();
    let q_half = ext_chip.mul_base_const(ctx, s2_minus_d3, half);
    let q = ext_chip.sub(ctx, q_half, s1);
    let p_plus_q = ext_chip.add(ctx, p, q);
    let r = ext_chip.sub(ctx, s1, p_plus_q);

    let p_mul_x = ext_chip.mul(ctx, p, *x);
    let px_plus_q = ext_chip.add(ctx, p_mul_x, q);
    let quad_mul_x = ext_chip.mul(ctx, px_plus_q, *x);
    let quad = ext_chip.add(ctx, quad_mul_x, r);
    let cubic = ext_chip.mul(ctx, quad, *x);
    ext_chip.add(ctx, cubic, *evals[0])
}

#[derive(Clone)]
struct ViewPairWire {
    local: BabyBearExtWire,
    next: BabyBearExtWire,
}

struct ConstraintEvaluatorWire<'a> {
    preprocessed: Option<&'a [ViewPairWire]>,
    partitioned_main: &'a [Vec<ViewPairWire>],
    is_first_row: BabyBearExtWire,
    is_last_row: BabyBearExtWire,
    public_values: &'a [BabyBearWire],
}

impl ConstraintEvaluatorWire<'_> {
    fn eval_var(
        &self,
        ctx: &mut Context<Fr>,
        ext_chip: &BabyBearExtChip,
        symbolic_var: SymbolicVariable<RootF>,
    ) -> BabyBearExtWire {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => {
                let value = &self.preprocessed.unwrap()[index];
                if offset == 0 {
                    value.local
                } else {
                    value.next
                }
            }
            Entry::Main { part_index, offset } => {
                let value = &self.partitioned_main[part_index][index];
                if offset == 0 {
                    value.local
                } else {
                    value.next
                }
            }
            Entry::Public => {
                let value = self.public_values[index];
                ext_chip.from_base_var(ctx, value)
            }
            _ => panic!("invalid constraint"),
        }
    }
}

fn eval_symbolic_nodes_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    evaluator: &ConstraintEvaluatorWire<'_>,
    nodes: &[SymbolicExpressionNode<RootF>],
) -> Vec<BabyBearExtWire> {
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut exprs: Vec<BabyBearExtWire> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let expr = match node {
            SymbolicExpressionNode::Variable(var) => evaluator.eval_var(ctx, ext_chip, *var),
            SymbolicExpressionNode::Constant(c) => ext_chip.from_base_const(ctx, *c),
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            } => ext_chip.add(ctx, exprs[*left_idx], exprs[*right_idx]),
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => ext_chip.sub(ctx, exprs[*left_idx], exprs[*right_idx]),
            SymbolicExpressionNode::Neg { idx, .. } => ext_chip.neg(ctx, exprs[*idx]),
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => ext_chip.mul(ctx, exprs[*left_idx], exprs[*right_idx]),
            SymbolicExpressionNode::IsFirstRow => evaluator.is_first_row,
            SymbolicExpressionNode::IsLastRow => evaluator.is_last_row,
            SymbolicExpressionNode::IsTransition => ext_chip.sub(ctx, one, evaluator.is_last_row),
        };
        exprs.push(expr);
    }
    exprs
}

fn column_openings_by_rot_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    openings: &[BabyBearExtWire],
    need_rot: bool,
) -> Vec<ViewPairWire> {
    shared_math::column_openings_by_rot_assigned(ctx, ext_chip, openings, need_rot)
        .into_iter()
        .map(|(local, next)| ViewPairWire { local, next })
        .collect::<Vec<_>>()
}

fn observe_layer_claims_assigned(
    ctx: &mut Context<Fr>,
    transcript: &mut TranscriptGadget,
    baby_bear: &BabyBearChip,
    claims: &[BabyBearExtWire],
) {
    for claim in claims {
        transcript.observe_ext(ctx, baby_bear, claim);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn constrain_batch_from_proof_inputs(
    ctx: &mut Context<Fr>,
    range: &halo2_base::gates::range::RangeChip<Fr>,
    transcript: &mut TranscriptGadget,
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    gkr_wire: &GkrProofWire,
    batch_wire: &BatchConstraintProofWire,
    n_per_trace: &[isize],
    trace_id_to_air_id: &[usize],
    public_values: Vec<Vec<BabyBearWire>>,
) -> BatchConstraintWire {
    let base_chip = Arc::new(BabyBearChip::new(Arc::new(range.clone())));
    let ext_chip = BabyBearExtChip::new(base_chip);
    let baby_bear = ext_chip.base();

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
    let n_logup_host = calculate_n_logup(l_skip, total_interactions_host);
    let n_max_host = n_per_trace.iter().copied().max().unwrap_or(0).max(0) as usize;
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
    transcript.check_witness(ctx, baby_bear, logup_pow_bits, &logup_pow_witness);

    let alpha_logup = transcript.sample_ext(ctx, baby_bear);
    let beta_logup = transcript.sample_ext(ctx, baby_bear);

    let gkr_q0_claim = (total_interactions_host != 0).then_some(gkr_wire.q0_claim);
    let gkr_claims_per_layer = &gkr_wire.claims_per_layer;
    let gkr_sumcheck_polys = &gkr_wire.sumcheck_polys;

    let zero = ext_chip.zero(ctx);
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let total_gkr_rounds = l_skip + n_logup_host;
    let (mut gkr_p_xi_claim, mut gkr_q_xi_claim, gkr_xi_claims, _gkr_sample_stream) =
        if total_interactions_host == 0 {
            let q0_claim = gkr_wire.q0_claim;
            ext_chip.assert_equal(ctx, q0_claim, one);
            (zero, alpha_logup, Vec::new(), Vec::new())
        } else {
            let q0_claim =
                gkr_q0_claim.expect("non-zero interaction branch must include q0 claim witness");
            transcript.observe_ext(ctx, baby_bear, &q0_claim);

            let layer0 = &gkr_claims_per_layer[0];
            observe_layer_claims_assigned(ctx, transcript, baby_bear, layer0);

            let p0_q1 = ext_chip.mul(ctx, layer0[0], layer0[3]);
            let p1_q0 = ext_chip.mul(ctx, layer0[2], layer0[1]);
            let p_cross = ext_chip.add(ctx, p0_q1, p1_q0);
            let q_cross = ext_chip.mul(ctx, layer0[1], layer0[3]);
            ext_chip.assert_equal(ctx, p_cross, zero);
            ext_chip.assert_equal(ctx, q_cross, q0_claim);

            let mu0 = transcript.sample_ext(ctx, baby_bear);
            let mut gkr_sample_stream = vec![mu0];
            let mut numer_claim =
                interpolate_linear_at_01_assigned(ctx, &ext_chip, &layer0[0], &layer0[2], &mu0);
            let mut denom_claim =
                interpolate_linear_at_01_assigned(ctx, &ext_chip, &layer0[1], &layer0[3], &mu0);
            let mut gkr_r = vec![mu0];

            for round in 1..total_gkr_rounds {
                let lambda_round = transcript.sample_ext(ctx, baby_bear);
                gkr_sample_stream.push(lambda_round);

                let lambda_denom = ext_chip.mul(ctx, lambda_round, denom_claim);
                let mut claim = ext_chip.add(ctx, numer_claim, lambda_denom);
                let round_polys = &gkr_sumcheck_polys[round - 1];
                let mut gkr_r_prime = Vec::with_capacity(round);
                let mut eq = one;

                for (subround, xi_prev) in gkr_r.iter().enumerate().take(round) {
                    let ev1 = round_polys[subround * 3];
                    let ev2 = round_polys[subround * 3 + 1];
                    let ev3 = round_polys[subround * 3 + 2];
                    transcript.observe_ext(ctx, baby_bear, &ev1);
                    transcript.observe_ext(ctx, baby_bear, &ev2);
                    transcript.observe_ext(ctx, baby_bear, &ev3);

                    let ri = transcript.sample_ext(ctx, baby_bear);
                    gkr_sample_stream.push(ri);
                    gkr_r_prime.push(ri);

                    let ev0 = ext_chip.sub(ctx, claim, ev1);
                    claim = interpolate_cubic_at_0123_assigned(
                        ctx,
                        &ext_chip,
                        [&ev0, &ev1, &ev2, &ev3],
                        &ri,
                    );
                    let xi_ri = ext_chip.mul(ctx, *xi_prev, ri);
                    let one_minus_xi = ext_chip.sub(ctx, one, *xi_prev);
                    let one_minus_ri = ext_chip.sub(ctx, one, ri);
                    let one_minus_term = ext_chip.mul(ctx, one_minus_xi, one_minus_ri);
                    let eq_factor = ext_chip.add(ctx, xi_ri, one_minus_term);
                    eq = ext_chip.mul(ctx, eq, eq_factor);
                }

                let layer_claims = &gkr_claims_per_layer[round];
                observe_layer_claims_assigned(ctx, transcript, baby_bear, layer_claims);

                let p0_q1 = ext_chip.mul(ctx, layer_claims[0], layer_claims[3]);
                let p1_q0 = ext_chip.mul(ctx, layer_claims[2], layer_claims[1]);
                let p_cross = ext_chip.add(ctx, p0_q1, p1_q0);
                let q_cross = ext_chip.mul(ctx, layer_claims[1], layer_claims[3]);
                let lambda_q_cross = ext_chip.mul(ctx, lambda_round, q_cross);
                let claim_sum = ext_chip.add(ctx, p_cross, lambda_q_cross);
                let expected_claim = ext_chip.mul(ctx, claim_sum, eq);
                ext_chip.assert_equal(ctx, expected_claim, claim);

                let mu_round = transcript.sample_ext(ctx, baby_bear);
                gkr_sample_stream.push(mu_round);
                numer_claim = interpolate_linear_at_01_assigned(
                    ctx,
                    &ext_chip,
                    &layer_claims[0],
                    &layer_claims[2],
                    &mu_round,
                );
                denom_claim = interpolate_linear_at_01_assigned(
                    ctx,
                    &ext_chip,
                    &layer_claims[1],
                    &layer_claims[3],
                    &mu_round,
                );
                gkr_r = core::iter::once(mu_round)
                    .chain(gkr_r_prime.into_iter())
                    .collect();
            }

            (numer_claim, denom_claim, gkr_r, gkr_sample_stream)
        };

    let mut xi = gkr_xi_claims;
    while xi.len() != l_skip + n_global_host {
        xi.push(transcript.sample_ext(ctx, baby_bear));
    }

    let lambda = transcript.sample_ext(ctx, baby_bear);

    let numerator_term_per_air = &batch_wire.numerator_term_per_air;
    let denominator_term_per_air = &batch_wire.denominator_term_per_air;
    for (num_term, den_term) in numerator_term_per_air
        .iter()
        .zip(denominator_term_per_air.iter())
    {
        gkr_p_xi_claim = ext_chip.sub(ctx, gkr_p_xi_claim, *num_term);
        gkr_q_xi_claim = ext_chip.sub(ctx, gkr_q_xi_claim, *den_term);
        transcript.observe_ext(ctx, baby_bear, num_term);
        transcript.observe_ext(ctx, baby_bear, den_term);
    }
    let gkr_numerator_residual = gkr_p_xi_claim;
    let gkr_denominator_claim = gkr_q_xi_claim;
    let gkr_denominator_residual = ext_chip.sub(ctx, gkr_denominator_claim, alpha_logup);
    ext_chip.assert_equal(ctx, gkr_numerator_residual, zero);
    ext_chip.assert_equal(ctx, gkr_denominator_residual, zero);

    let mu = transcript.sample_ext(ctx, baby_bear);

    let mut sum_claim = ext_chip.zero(ctx);
    let mut cur_mu_pow = one;
    for (num_term, den_term) in numerator_term_per_air
        .iter()
        .zip(denominator_term_per_air.iter())
    {
        let num_weighted = ext_chip.mul(ctx, *num_term, cur_mu_pow);
        sum_claim = ext_chip.add(ctx, sum_claim, num_weighted);
        cur_mu_pow = ext_chip.mul(ctx, cur_mu_pow, mu);

        let den_weighted = ext_chip.mul(ctx, *den_term, cur_mu_pow);
        sum_claim = ext_chip.add(ctx, sum_claim, den_weighted);
        cur_mu_pow = ext_chip.mul(ctx, cur_mu_pow, mu);
    }

    let univariate_round_coeffs = &batch_wire.univariate_round_coeffs;
    for coeff in univariate_round_coeffs {
        transcript.observe_ext(ctx, baby_bear, coeff);
    }
    let mut r = vec![transcript.sample_ext(ctx, baby_bear)];

    let stride = 1usize << l_skip;
    let mut sum_univ_domain_s_0 = ext_chip.zero(ctx);
    for coeff in univariate_round_coeffs.iter().step_by(stride) {
        sum_univ_domain_s_0 = ext_chip.add(ctx, sum_univ_domain_s_0, *coeff);
    }
    let sum_univ_domain_s_0 =
        ext_chip.mul_base_const(ctx, sum_univ_domain_s_0, RootF::from_u64(stride as u64));
    ext_chip.assert_equal(ctx, sum_claim, sum_univ_domain_s_0);

    let sumcheck_round_polys = &batch_wire.sumcheck_round_polys;
    let mut consistency_lhs = eval_ext_poly_horner(ctx, &ext_chip, univariate_round_coeffs, &r[0]);
    for round_evals in sumcheck_round_polys {
        for eval in round_evals {
            transcript.observe_ext(ctx, baby_bear, eval);
        }

        let s_1 = round_evals[0];
        let s_0 = ext_chip.sub(ctx, consistency_lhs, s_1);
        let mut interpolation_evals = Vec::with_capacity(round_evals.len() + 1);
        interpolation_evals.push(s_0);
        interpolation_evals.extend(round_evals.iter().copied());
        let next_r = transcript.sample_ext(ctx, baby_bear);
        consistency_lhs =
            eval_lagrange_on_integer_grid(ctx, &ext_chip, &next_r, &interpolation_evals);
        r.push(next_r);
    }

    let column_openings = &batch_wire.column_openings;

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let need_rot = column_openings_need_rot[trace_idx][0];
        for claim in column_openings_by_rot_assigned(ctx, &ext_chip, &air_openings[0], need_rot) {
            transcript.observe_ext(ctx, baby_bear, &claim.local);
            transcript.observe_ext(ctx, baby_bear, &claim.next);
        }
    }

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        for (part_idx, claims) in air_openings.iter().enumerate().skip(1) {
            let need_rot = column_openings_need_rot[trace_idx][part_idx];
            for claim in column_openings_by_rot_assigned(ctx, &ext_chip, claims, need_rot) {
                transcript.observe_ext(ctx, baby_bear, &claim.local);
                transcript.observe_ext(ctx, baby_bear, &claim.next);
            }
        }
    }

    let mut eq_3b_per_trace = Vec::with_capacity(n_per_trace.len());
    let mut stacked_idx = 0usize;
    for (trace_idx, &n) in n_per_trace.iter().enumerate() {
        let n_lift = n.max(0) as usize;
        let interactions = &trace_interactions[trace_idx];
        if interactions.is_empty() {
            eq_3b_per_trace.push(Vec::new());
            continue;
        }

        let mut eq_3b = Vec::with_capacity(interactions.len());
        for _ in 0..interactions.len() {
            let mut b_int = stacked_idx >> (l_skip + n_lift);
            let mut b_vec = Vec::with_capacity(n_logup_host.saturating_sub(n_lift));
            for _ in 0..n_logup_host.saturating_sub(n_lift) {
                b_vec.push((b_int & 1) == 1);
                b_int >>= 1;
            }
            stacked_idx += 1 << (l_skip + n_lift);
            let eq = eval_eq_mle_binary_assigned(
                ctx,
                &ext_chip,
                &xi[l_skip + n_lift..l_skip + n_logup_host],
                &b_vec,
            );
            eq_3b.push(eq);
        }
        eq_3b_per_trace.push(eq_3b);
    }

    let mut eq_ns = vec![one; n_max_host + 1];
    let mut eq_sharp_ns = vec![one; n_max_host + 1];
    eq_ns[0] = eval_eq_uni_assigned(ctx, &ext_chip, l_skip, &xi[0], &r[0]);
    eq_sharp_ns[0] =
        eval_eq_sharp_uni_assigned(ctx, &ext_chip, &omega_skip_pows, &xi[..l_skip], &r[0]);
    for (i, r_i) in r.iter().enumerate().skip(1) {
        let eq_mle = eval_eq_mle_assigned(
            ctx,
            &ext_chip,
            &[xi[l_skip + i - 1]],
            core::slice::from_ref(r_i),
        );
        eq_ns[i] = ext_chip.mul(ctx, eq_ns[i - 1], eq_mle);
        eq_sharp_ns[i] = ext_chip.mul(ctx, eq_sharp_ns[i - 1], eq_mle);
    }
    if n_max_host > 0 {
        let n_max_usize = n_max_host;
        let mut r_rev_prod = r[n_max_usize];
        for i in (0..n_max_usize).rev() {
            eq_ns[i] = ext_chip.mul(ctx, eq_ns[i], r_rev_prod);
            eq_sharp_ns[i] = ext_chip.mul(ctx, eq_sharp_ns[i], r_rev_prod);
            r_rev_prod = ext_chip.mul(ctx, r_rev_prod, r[i]);
        }
    }

    let mut interactions_evals = Vec::new();
    let mut constraints_evals = Vec::new();
    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let air_idx = trace_id_to_air_id_host[trace_idx];
        let n = n_per_trace[trace_idx];
        let n_lift = n.max(0) as usize;

        let need_rot_flags = &column_openings_need_rot[trace_idx];
        let common_main =
            column_openings_by_rot_assigned(ctx, &ext_chip, &air_openings[0], need_rot_flags[0]);
        let has_preprocessed = trace_has_preprocessed[trace_idx];
        let preprocessed = has_preprocessed.then(|| {
            column_openings_by_rot_assigned(ctx, &ext_chip, &air_openings[1], need_rot_flags[1])
        });
        let cached_idx = 1 + has_preprocessed as usize;
        let mut partitioned_main = air_openings[cached_idx..]
            .iter()
            .enumerate()
            .map(|(part_offset, opening)| {
                column_openings_by_rot_assigned(
                    ctx,
                    &ext_chip,
                    opening,
                    need_rot_flags[cached_idx + part_offset],
                )
            })
            .collect::<Vec<_>>();
        partitioned_main.push(common_main);

        let (l, rs_n, norm_factor) = if n.is_negative() {
            (
                l_skip.wrapping_add_signed(n),
                vec![ext_chip.pow_power_of_two(ctx, r[0], n.unsigned_abs())],
                RootF::from_usize(1usize << n.unsigned_abs()).inverse(),
            )
        } else {
            (l_skip, r[..=n_lift].to_vec(), RootF::ONE)
        };

        let inv_l = RootF::from_usize(1usize << l).inverse();
        let mut is_first_row = progression_exp_2_assigned(ctx, &ext_chip, &rs_n[0], l);
        is_first_row = ext_chip.mul_base_const(ctx, is_first_row, inv_l);
        for x in rs_n.iter().skip(1) {
            let one_minus_x = ext_chip.sub(ctx, one, *x);
            is_first_row = ext_chip.mul(ctx, is_first_row, one_minus_x);
        }

        let omega = RootF::two_adic_generator(l);
        let rs0_omega = ext_chip.mul_base_const(ctx, rs_n[0], omega);
        let mut is_last_row = progression_exp_2_assigned(ctx, &ext_chip, &rs0_omega, l);
        is_last_row = ext_chip.mul_base_const(ctx, is_last_row, inv_l);
        for x in rs_n.iter().skip(1) {
            is_last_row = ext_chip.mul(ctx, is_last_row, *x);
        }

        let evaluator = ConstraintEvaluatorWire {
            preprocessed: preprocessed.as_deref(),
            partitioned_main: &partitioned_main,
            is_first_row,
            is_last_row,
            public_values: public_values[air_idx].as_slice(),
        };

        let node_values = eval_symbolic_nodes_assigned(
            ctx,
            &ext_chip,
            &evaluator,
            &trace_constraint_nodes[trace_idx],
        );

        let mut expr = ext_chip.zero(ctx);
        let mut lambda_pow = one;
        for &constraint_idx in &trace_constraint_indices[trace_idx] {
            let term = ext_chip.mul(ctx, node_values[constraint_idx], lambda_pow);
            expr = ext_chip.add(ctx, expr, term);
            lambda_pow = ext_chip.mul(ctx, lambda_pow, lambda);
        }
        constraints_evals.push(ext_chip.mul(ctx, eq_ns[n_lift], expr));

        let interactions = &trace_interactions[trace_idx];
        let eq_3bs = &eq_3b_per_trace[trace_idx];
        let mut num = ext_chip.zero(ctx);
        let mut denom = ext_chip.zero(ctx);
        for (eq_3b, interaction) in eq_3bs.iter().zip(interactions.iter()) {
            let count_eval = node_values[interaction.count];
            let mut denom_eval = ext_chip.zero(ctx);
            let mut beta_pow = one;
            for &msg_idx in &interaction.message {
                let term = ext_chip.mul(ctx, node_values[msg_idx], beta_pow);
                denom_eval = ext_chip.add(ctx, denom_eval, term);
                beta_pow = ext_chip.mul(ctx, beta_pow, beta_logup);
            }
            let bus_const = ext_chip
                .from_base_const(ctx, RootF::from_u64(u64::from(interaction.bus_index) + 1));
            let bus_term = ext_chip.mul(ctx, bus_const, beta_pow);
            denom_eval = ext_chip.add(ctx, denom_eval, bus_term);

            let eq_times_count = ext_chip.mul(ctx, *eq_3b, count_eval);
            num = ext_chip.add(ctx, num, eq_times_count);
            let eq_times_denom = ext_chip.mul(ctx, *eq_3b, denom_eval);
            denom = ext_chip.add(ctx, denom, eq_times_denom);
        }

        let norm = ext_chip.from_base_const(ctx, norm_factor);
        let num_norm = ext_chip.mul(ctx, num, norm);
        let num_scaled = ext_chip.mul(ctx, num_norm, eq_sharp_ns[n_lift]);
        let denom_scaled = ext_chip.mul(ctx, denom, eq_sharp_ns[n_lift]);
        interactions_evals.push(num_scaled);
        interactions_evals.push(denom_scaled);
    }

    let mut consistency_rhs = ext_chip.zero(ctx);
    let mut cur_mu_pow = one;
    for term in interactions_evals.iter().chain(constraints_evals.iter()) {
        let weighted_term = ext_chip.mul(ctx, *term, cur_mu_pow);
        consistency_rhs = ext_chip.add(ctx, consistency_rhs, weighted_term);
        cur_mu_pow = ext_chip.mul(ctx, cur_mu_pow, mu);
    }
    let consistency_residual = ext_chip.sub(ctx, consistency_lhs, consistency_rhs);
    ext_chip.assert_equal(ctx, consistency_residual, zero);

    BatchConstraintWire {
        column_openings: column_openings.clone(),
        r,
    }
}
