use halo2curves_axiom::bn256::{G1Affine, G2Affine};
use rand::{rngs::StdRng, SeedableRng};

use crate::common::EcPoint;

#[test]
#[allow(non_snake_case)]
fn test_multi_miller_loop_bn254() {
    // Generate random G1 and G2 points
    let mut rng0 = StdRng::seed_from_u64(8);
    let rnd_pt0 = G1Affine::random(&mut rng0);
    let P = EcPoint {
        x: rnd_pt0.x,
        y: rnd_pt0.y,
    };
    let mut rng1 = StdRng::seed_from_u64(8 * 2);
    let rnd_pt1 = G2Affine::random(&mut rng1);
    let Q = EcPoint {
        x: rnd_pt1.x,
        y: rnd_pt1.y,
    };
    println!("{:#?}", P);
    println!("{:#?}", Q);

    // // halo2curves pseudo-binary encoding
    // let pbe = SIX_U_PLUS_2_NAF
    //     .iter()
    //     .map(|&x| x as i32)
    //     .collect::<Vec<i32>>();
    // let pbe = pbe.as_slice();
    // println!("{:?}", pbe);

    // Run the multi-miller loop
}
