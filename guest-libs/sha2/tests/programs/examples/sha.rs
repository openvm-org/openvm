#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use hex::FromHex;
use openvm_sha2::sha256;

openvm::entry!(main);

pub fn main() {
    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ("98c1c0bdb7d5fea9a88859f06c6c439f", "b6b2c9c9b6f30e5c66c977f1bd7ad97071bee739524aecf793384890619f2b05"),
        ("5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c", "ac0e25049870b91d78ef6807bb87fce4603c81abd3c097fba2403fd18b6ce0b7"),
        ("9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e", "080ad71239852124fc26758982090611b9b19abf22d22db3a57f67a06e984a23")
    ];
    for (input, expected_output) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected_output = Vec::from_hex(expected_output).unwrap();
        let output = sha256(&black_box(input));
        if output != *expected_output {
            panic!();
        }
    }
}
