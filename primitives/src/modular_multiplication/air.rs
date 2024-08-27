use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use p3_field::AbstractField;

use crate::modular_multiplication::{
    columns::ModularMultiplicationCols, FullLimbs, LimbDimensions,
};

pub fn range_check<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: usize,
    decomp: usize,
    bits: usize,
    into_expr: impl Into<AB::Expr>,
) {
    assert!(bits <= decomp);
    let expr = into_expr.into();
    if bits == decomp {
        builder.push_send(range_bus, [expr], AB::F::one());
    } else {
        builder.push_send(range_bus, [expr.clone()], AB::F::one());
        builder.push_send(
            range_bus,
            [expr + AB::F::from_canonical_usize((1 << decomp) - (1 << bits))],
            AB::F::one(),
        );
    }
}

pub fn constrain_limbs<AB: InteractionBuilder>(
    builder: &mut AB,
    range_bus: usize,
    decomp: usize,
    limb_dimensions: &LimbDimensions,
    local: ModularMultiplicationCols<AB::Var>,
) -> FullLimbs<AB::Expr> {
    let ModularMultiplicationCols { io, aux } = local;

    let [a_limbs, b_limbs, r_limbs] = [
        (io.a_elems, &aux.a_limbs_without_first),
        (io.b_elems, &aux.b_limbs_without_first),
        (io.r_elems, &aux.r_limbs_without_first),
    ]
    .map(|(elems, limbs_without_first)| {
        limb_dimensions
            .io_limb_sizes
            .iter()
            .zip_eq(elems.iter().zip_eq(limbs_without_first))
            .map(|(limb_sizes, (&elem, limbs_here_without_first))| {
                let mut first_limb = elem.into();
                let mut shift = limb_sizes[0];
                for (&limb_size, &limb) in
                    limb_sizes.iter().skip(1).zip_eq(limbs_here_without_first)
                {
                    first_limb -= AB::Expr::from_canonical_usize(1 << shift) * limb;
                    shift += limb_size;
                }
                let mut limbs = vec![first_limb];
                limbs.extend(limbs_here_without_first.iter().map(|&limb| limb.into()));
                for (&limb_size, limb) in limb_sizes.iter().zip_eq(&limbs) {
                    range_check(builder, range_bus, decomp, limb_size, limb.clone());
                }
                limbs
            })
            .collect_vec()
    });

    for (&limb_size, &limb) in limb_dimensions.q_limb_sizes.iter().zip_eq(&aux.q_limbs) {
        range_check(builder, range_bus, decomp, limb_size, limb);
    }

    FullLimbs {
        a_limbs,
        b_limbs,
        r_limbs,
        q_limbs: aux.q_limbs.iter().map(|&limb| limb.into()).collect(),
    }
}
