#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use hex::FromHex;
use openvm_sha2::{Sha256, Sha384, Sha512};

openvm::entry!(main);

struct ShaTestVector {
    input: &'static str,
    expected_output_sha256: &'static str,
    expected_output_sha512: &'static str,
    expected_output_sha384: &'static str,
}

pub fn main() {
    let test_vectors = [
        ShaTestVector {
            input: "",
            expected_output_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            expected_output_sha512: "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
            expected_output_sha384: "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
        },
        ShaTestVector {
            input: "98c1c0bdb7d5fea9a88859f06c6c439f",
            expected_output_sha256: "b6b2c9c9b6f30e5c66c977f1bd7ad97071bee739524aecf793384890619f2b05",
            expected_output_sha512: "eb576959c531f116842c0cc915a29c8f71d7a285c894c349b83469002ef093d51f9f14ce4248488bff143025e47ed27c12badb9cd43779cb147408eea062d583",
            expected_output_sha384: "63e3061aab01f335ea3a4e617b9d14af9b63a5240229164ee962f6d5335ff25f0f0bf8e46723e83c41b9d17413b6a3c7",
        },
        ShaTestVector {
            input: "5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c",
            expected_output_sha256: "ac0e25049870b91d78ef6807bb87fce4603c81abd3c097fba2403fd18b6ce0b7",
            expected_output_sha512: "a20d5fb14814d045a7d2861e80d2b688f1cd1daaba69e6bb1cc5233f514141ea4623b3373af702e78e3ec5dc8c1b716a37a9a2f5fbc9493b9df7043f5e99a8da",
            expected_output_sha384: "eac4b72b0540486bc088834860873338e31e9e4062532bf509191ef63b9298c67db5654a28fe6f07e4cc6ff466d1be24",
        },
        ShaTestVector {
            input: "9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e",
            expected_output_sha256: "080ad71239852124fc26758982090611b9b19abf22d22db3a57f67a06e984a23",
            expected_output_sha512: "8d215ee6dc26757c210db0dd00c1c6ed16cc34dbd4bb0fa10c1edb6b62d5ab16aea88c881001b173d270676daf2d6381b5eab8711fa2f5589c477c1d4b84774f",
            expected_output_sha384: "904a90010d772a904a35572fdd4bdf1dd253742e47872c8a18e2255f66fa889e44781e65487a043f435daa53c496a53e",
        }
    ];

    for (
        i,
        ShaTestVector {
            input,
            expected_output_sha256,
            expected_output_sha512,
            expected_output_sha384,
        },
    ) in test_vectors.iter().enumerate()
    {
        let input = Vec::from_hex(input).unwrap();
        let expected_output_sha256 = Vec::from_hex(expected_output_sha256).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(black_box(&input));
        let output = hasher.finalize();
        if output != *expected_output_sha256 {
            panic!(
                "sha256 test {i} failed on input: {:?}.\nexpected: {:?},\ngot: {:?}",
                input, expected_output_sha256, output
            );
        }
        let expected_output_sha512 = Vec::from_hex(expected_output_sha512).unwrap();
        let mut hasher = Sha512::new();
        hasher.update(black_box(&input));
        let output = hasher.finalize();
        if output != *expected_output_sha512 {
            panic!(
                "sha512 test {i} failed on input: {:?}.\nexpected: {:?},\ngot: {:?}",
                input, expected_output_sha512, output
            );
        }
        let expected_output_sha384 = Vec::from_hex(expected_output_sha384).unwrap();
        let mut hasher = Sha384::new();
        hasher.update(black_box(&input));
        let output = hasher.finalize();
        if output != *expected_output_sha384 {
            panic!(
                "sha384 test {i} failed on input: {:?}.\nexpected: {:?},\ngot: {:?}",
                input, expected_output_sha384, output
            );
        }
    }
}
