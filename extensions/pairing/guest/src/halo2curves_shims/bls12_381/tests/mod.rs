use alloc::vec::Vec;

use halo2curves_axiom::bls12_381::{Fq, Fq2, G1Affine, G2Affine};
use itertools::izip;
use openvm_ecc_guest::AffinePoint;
use rand::{rngs::StdRng, SeedableRng};

#[cfg(test)]
mod test_final_exp;
#[cfg(test)]
mod test_line;
#[cfg(test)]
mod test_miller_loop;

#[cfg(not(target_os = "zkvm"))]
#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn generate_test_points_bls12_381(
    rand_seeds: &[u64],
) -> (
    Vec<G1Affine>,
    Vec<G2Affine>,
    Vec<AffinePoint<Fq>>,
    Vec<AffinePoint<Fq2>>,
) {
    let (P_vec, Q_vec) = rand_seeds
        .iter()
        .map(|seed| {
            let mut rng0 = StdRng::seed_from_u64(*seed);
            let p = G1Affine::random(&mut rng0);
            let mut rng1 = StdRng::seed_from_u64(*seed * 2);
            let q = G2Affine::random(&mut rng1);
            (p, q)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let (P_ecpoints, Q_ecpoints) = izip!(P_vec.clone(), Q_vec.clone())
        .map(|(P, Q)| {
            (
                AffinePoint { x: P.x, y: P.y },
                AffinePoint { x: Q.x, y: Q.y },
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    (P_vec, Q_vec, P_ecpoints, Q_ecpoints)
}
