use axvm_ecc::{pairing::MultiMillerLoop, AffinePoint};
use halo2curves_axiom::{
    bn256::{Fq, Fq2, G1Affine, G2Affine, G2Prepared, Gt},
    pairing::MillerLoopResult,
};

use crate::{curves::bn254::Bn254, tests::utils::generate_test_points};

#[allow(non_snake_case)]
fn run_miller_loop_test(rand_seeds: &[u64]) {
    let (P_vec, Q_vec, P_ecpoints, Q_ecpoints) =
        generate_test_points::<G1Affine, G2Affine, Fq, Fq2>(rand_seeds);

    // Compare against halo2curves implementation
    let g2_prepareds = Q_vec
        .iter()
        .map(|q| G2Prepared::from(*q))
        .collect::<Vec<_>>();
    let terms = P_vec.iter().zip(g2_prepareds.iter()).collect::<Vec<_>>();
    let compare_miller = halo2curves_axiom::bn256::multi_miller_loop(terms.as_slice());
    let compare_final = compare_miller.final_exponentiation();

    // Run the multi-miller loop
    let f = Bn254::multi_miller_loop(P_ecpoints.as_slice(), Q_ecpoints.as_slice());

    let wrapped_f = Gt(f);
    let final_f = wrapped_f.final_exponentiation();

    // Run halo2curves final exponentiation on our multi_miller_loop output
    assert_eq!(final_f, compare_final);
}

#[test]
#[allow(non_snake_case)]
fn test_single_miller_loop_bn254() {
    let rand_seeds = [925];
    run_miller_loop_test(&rand_seeds);
}

#[test]
#[allow(non_snake_case)]
fn test_multi_miller_loop_bn254() {
    let rand_seeds = [8, 15, 29, 55, 166];
    run_miller_loop_test(&rand_seeds);
}

#[test]
fn test_multi_miller_loop_generators() {
    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    let p = AffinePoint { x: g1.x, y: g1.y };
    let q = AffinePoint { x: g2.x, y: g2.y };

    let f = Bn254::multi_miller_loop(&[p], &[q]);
    println!("final f: {:#?}", f);
}

#[test]
fn test_fq_display() {
    // let fq = Fq([
    //     0xf32cfc5b538afa89,
    //     0xb5e71911d44501fb,
    //     0x47ab1eff0a417ff6,
    //     0x06d89f71cab8351f,
    // ]);
    // println!(
    //     "fq_raw: {:x?}, {:x?}, {:x?}, {:x?}",
    //     fq.0[0], fq.0[1], fq.0[2], fq.0[3]
    // );
    // println!("fq: {:?}", fq);
    println!(
        "hex_rev: {:?}",
        hex_rev("0x16c9e55061ebae204ba4cc8bd75a079432ae2a1d0b7c9dce1665d51c640fcba2")
    );

    println!(
        "hex_rev: {:?}",
        hex_rev("0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48")
    );

    println!(
        "hex_rev: {:?}",
        hex_rev("0x063cf305489af5dcdc5ec698b6e2f9b9dbaae0eda9c95998dc54014671a0135a")
    );

    println!(
        "hex_rev: {:?}",
        hex_rev("0x07c03cbcac41049a0704b5a7ec796f2b21807dc98fa25bd282d37f632623b0e3")
    );
}

fn hex_rev(input: &str) -> String {
    // Remove "0x" prefix if present
    let cleaned = input.trim_start_matches("0x");

    // Convert to bytes in pairs and reverse
    cleaned
        .chars()
        .collect::<Vec<char>>()
        .chunks(2)
        .map(|chunk| chunk.iter().collect::<String>())
        .rev()
        .collect::<Vec<String>>()
        .join("")
}

#[test]
fn to_le_hex_string() {
    use num_bigint::BigUint;
    use num_traits::Num;
    let hex_vals = [
        // [0]
        [
            "d35d438dc58f0d9d",
            "0a78eb28f5c70b3d",
            "666ea36f7879462c",
            "0e0a77c19a07df2f",
        ],
        // [1]
        [
            "b5773b104563ab30",
            "347f91c8a9aa6454",
            "7a007127242e0991",
            "1956bcd8118214ec",
        ],
        [
            "6e849f1ea0aa4757",
            "aa1c7b6d89f89141",
            "b6e713cdfae0ca3a",
            "26694fbb4e82ebc3",
        ],
        // [2]
        [
            "3350c88e13e80b9c",
            "7dce557cdb5e56b9",
            "6001b4b8b615564a",
            "2682e617020217e0",
        ],
        // xi_to_q_minus_1_over_2
        [
            "e4bbdd0c2936b629",
            "bb30f162e133bacb",
            "31a9d1b6f9645366",
            "253570bea500f8dd",
        ],
        [
            "a1d77ce45ffe77c7",
            "07affd117826d1db",
            "6d16bd27bb7edc6b",
            "2c87200285defecc",
        ],
    ];

    let hex_le_strs: Vec<String> = hex_vals
        .iter()
        .map(|fq_parts| {
            let concatenated = fq_parts
                .iter()
                .rev() // Reverse the order of the 4 parts
                .map(|s| {
                    let value = u64::from_str_radix(s, 16).unwrap();
                    format!("{:016x}", value)
                })
                .collect::<String>();

            // Convert the concatenated string to bytes (2 chars per byte)
            concatenated
                .as_bytes()
                .chunks(2)
                .rev() // Reverse the byte order
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect::<String>()
        })
        .collect();

    // let hex_le_strs: Vec<String> = hex_vals
    //     .iter()
    //     .map(|fq_parts| {
    //         // Reverse each part and concatenate
    //         fq_parts
    //             .iter()
    //             .rev() // Reverse the order of the 4 parts for little-endian
    //             .map(|s| {
    //                 // Parse each hex string to ensure it's valid
    //                 let value = u64::from_str_radix(s, 16).unwrap();
    //                 // Format back to hex string, maintaining 16 characters (8 bytes) width
    //                 format!("{:016x}", value)
    //             })
    //             .collect::<String>()
    //     })
    //     .collect();

    for hex_le_str in hex_le_strs {
        println!("{:?}", hex_le_str);
    }
}
