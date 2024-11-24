use axvm::moduli_setup;
use axvm_algebra::{Field, IntMod};

mod fp12;
mod fp2;
pub mod pairing;

pub use fp12::*;
pub use fp2::*;
use hex_literal::hex;

use crate::pairing::PairingIntrinsics;

#[cfg(all(test, feature = "halo2curves", not(target_os = "zkvm")))]
mod tests;

pub const BN254_SEED: u64 = 0x44e992b44a6909f1;
pub const BN254_PSEUDO_BINARY_ENCODING: [i8; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

pub struct Bn254;

impl Bn254 {
    pub const FROBENIUS_COEFF_FQ6_C1: [Fp2; 3] = [
        Fp2 {
            c0: Bn254Fp(hex!(
                "9d0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e"
            )),
            c1: Bn254Fp(hex!(
                "0000000000000000000000000000000000000000000000000000000000000000"
            )),
        },
        Fp2 {
            c0: Bn254Fp(hex!(
                "3d556f175795e3990c33c3c210c38cb743b159f53cec0b4cf711794f9847b32f"
            )),
            c1: Bn254Fp(hex!(
                "a2cb0f641cd56516ce9d7c0b1d2aae3294075ad78bcca44b20aeeb6150e5c916"
            )),
        },
        Fp2 {
            c0: Bn254Fp(hex!(
                "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
            )),
            c1: Bn254Fp(hex!(
                "0000000000000000000000000000000000000000000000000000000000000000"
            )),
        },
    ];

    pub const XI_TO_Q_MINUS_1_OVER_2: Fp2 = Fp2 {
        c0: Bn254Fp(hex!(
            "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
        )),
        c1: Bn254Fp(hex!(
            "e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
        )),
    };

    pub const U27_COEFFS: [Fp; 2] = [
        Bn254Fp(hex!(
            "63b53231799490bfde5cefaed2782499b74ea212ef6728863f6583d5bb90f714"
        )),
        Bn254Fp(hex!(
            "65ccaa44fe93004c6dd127893b87c8441fceae491de3844ff696b102553e060a"
        )),
    ];

    pub const EXP1: [u8; 381] = hex!("02fae42e4973924a5d67a5d2596440dd3153897c68d516d062f8015f36c3fc152a73114859eb4cec3eff8e12f119e6b3b6373de8dfb0178e622c3aaf9212aff3b863cdce2f19e45304da0689789b99a5c06b47a88ae84ed08d6375382d5b43e8559e471ed9edafa93168993756f755226d1fb5139f370bd1df6206ad8eb5de835af494b06108d38b7277eb44eca38d36e079b3574999625bea8196289db44b431aae371e85680fa342f7100bba591589f02cf8ecb79868d7443cc60fe8f47559371ceb7cb43c85c4759d9070063939f77b19cd8e05a93bf3094d3d5518628d5a19e9c247d98c4be6bdd2b8801788ce9faea263de92133df7ebbc224e9753aa9ef14c0ae51fac5266c00ed4ded7ce442caa657048173fabe2a396c821ee1dc6e7821edb8a5c6030fad438f6730455aa926463b3f1adb189f37c0ecd514e9f49699c7d2e3b274dcd3d267739953ce9f65a41395df713745bdaf039c199f69e6a6b1d46fc408cb6973b1dfda111a7c09504ce57838ff3eb46f64643825060");
    pub const EXP2: [u8; 380] = hex!("1c41570c33b015b79b8ae1dd628724a303180e16d87c1aae550988446612d16063139a622528c66125e8d14b4ccfa7d757611b9681c7fbcabf448b3e4c8b41ea6282e1b5e483b739118e76de8a80f32d0d746e89e2ad95f2460dd2ac58b6a9ca483b85e2253f6514cafc3174ffadd1de1d9e6905e5982437ae4c196419b2e86baa3e6558ed5d2b1733d94e67944799bc7649033b9d62752e9c80d218cdc9e60a783aee2added9dc01c1d09ea7513c2bd0b25a9f3d63cc59a3b23c7b34b6ca47d4cec46fd31a623b8811850003b02259c43537709b0ef41e3e66a915fb7cc58aba05c8766df452e78ceea638f1807a7196e4f8cfbedbfecbce10ac08a586e9d9728be087c6eadb7f42679a9704a028665203390e46aa78d2280d80141b540417698d8b945caac7b1157716c8e61fcd603b7d741d7069354deab370302e9747f2bb8c8d2c8911a907caf15187d83ecae026a6bc6c7d4e6f5255778f2bd483cd48f4e7b1ed5cddfacdb2f51d13f1a187b6ff0473e387102a0d331e861cb");
    pub const R_INV: [u8; 349] = hex!("2a71a42aee79815b691a6a294ef17857b96814f37b4c2fca934b0d67ba847f3dae4767238ba5fa4275e36247c2e0d2744ba5f7858382165fad421d6efacc9f2cef8aa8b968ee7b16d7001a28fd28206b1dd3e111f223575e4b046e455151acfb2c2b91395f068e0b9719edade8e326d9499c2238cac190b44c4eaa05ef379e03a9a5e228c5fe45c6ba38a2a7d2f8a2e6743a1e2509fd9bd55b098443f5d8252918fdce3f1e4dbb3c5f77a5e53d182418ed6397d5dadf7db6f0f880457bfe0b42fd3f9e4cb6b517826e8d0a7ea3026024e519498470b0c4571cbda9e9b9c3c33ad27ef8286dd457e4e53e766cdfcce86cb5c9a967d965cfacd3e5294c641f401dc214ffb6d3aa284fca9651ce25315372b248a33b03c647b55ab5455a1f1c4d0568f9a146bb7ebfe5d9a66b49bb00f5fd95b7c6784e70aac1f263098d89384a4274ed212efc2b410d5c2e47c1b4a5f0d2dcad831751a11b194379daaca1");
    pub const M_INV: [u8; 349] = hex!("0186f601e08dd1549b7e60deaa715424f40cb25186adc0eee93555b6d45d32122cd9a4aa34f544c6143b686a580bd4341f0e9cc796a29b3e0abf45fbcbe534278298cad6742d6eb7558005a1eedec7495884c546a95261846036719b2e8591195a1879549fce20f6fcdca29467df0ea52e1e84aa13e51a66ac9a83c506fd3d06751c3e97a391f3035f8757e549496f56776b19ac890d0f5585b4ca70cc9d001574cb7631fa8a10190cff16e8e4eebee1afc60f52092ddd7fea7f5d2359e82a55d5a62a0447b6fe4cd0382391724d90962b1fc6ee8fff1e01c9b766a835349070d501f354a4111b573624f772f393f8674a5444d2ec4ce226be79e0093e38188d68645ec977b3667969b57a1a59c55e1e6afd4f17a618fa924d8034a528b459b999d3ac8a7d51c8b0e8579cc52388c5e5a473bf4dc0072d7a403a1e679c843248b70bc58b9bf08969a9108c577fa41c1525e62392aff7721c7aff2f56b7");
}

moduli_setup! {
    Bn254Fp { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}

pub type Fp = Bn254Fp;

impl Field for Fp {
    type SelfRef<'a> = &'a Self;
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    fn double_assign(&mut self) {
        IntMod::double_assign(self);
    }

    fn square_assign(&mut self) {
        IntMod::square_assign(self);
    }
}

impl PairingIntrinsics for Bn254 {
    type Fp = Fp;
    type Fp2 = Fp2;
    type Fp12 = Fp12;

    const PAIRING_IDX: usize = 0;
    const XI: Fp2 = Fp2::new(Fp::from_const_u8(9), Fp::from_const_u8(1));
    const FROBENIUS_COEFFS: [[Self::Fp2; 5]; 12] = [
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "70e4c9dcda350bd676212f29081e525c608be676dd9fb9e8dfa765281cb78412"
                )),
                c1: Bn254Fp(hex!(
                    "ac62f3805ff05ccae5c7ee8e779279748e0b1512fe7c32a6e6e7fab4f3966924"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "3d556f175795e3990c33c3c210c38cb743b159f53cec0b4cf711794f9847b32f"
                )),
                c1: Bn254Fp(hex!(
                    "a2cb0f641cd56516ce9d7c0b1d2aae3294075ad78bcca44b20aeeb6150e5c916"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
                )),
                c1: Bn254Fp(hex!(
                    "e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "62a71e92551f8a8472ec94bef76533d3841e185ab7c0f38001a8ee645e4fb505"
                )),
                c1: Bn254Fp(hex!(
                    "26812bcd11473bc163c7de1bead28536921c0b3bb0803a9fee8afde7db5e142c"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "2f69b7ea10c8a22ed31baa559b455c42f43f35a461363ae94986794fe7c18301"
                )),
                c1: Bn254Fp(hex!(
                    "4b2c0c6eeeb8c624c02a8e6799cb80b07d9f72c746b27fa27506fd76caf2ac12"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "49fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ffffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "7fa6d41e397d6fe84ad255be8db34c8990aaacd08c60e9efbbe482cccf81dc19"
                )),
                c1: Bn254Fp(hex!(
                    "01c1c0f42baa9476ec39d497e3a5037f9d137635e3eecb06737de70bb6f8ab00"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "6dfbdc7be86e747bd342695d3dfd5f80ac259f95771cffba0aef55b778e05608"
                )),
                c1: Bn254Fp(hex!(
                    "de86a5aa2bab0c383126ff98bf31df0f4f0926ec6d0ef3a96f76d1b341def104"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ede9dc66d08acc5ff470a8bea389d6bba35e9eca1d7ff1db4caa96986d5b272a"
                )),
                c1: Bn254Fp(hex!(
                    "644c59b2b30c4db9ba6ecfd8c7ec007632e907950e904bb18f9bf034b611a428"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "66f0cb3cbc921a0ecb6bb075450933e64e44b2b5f7e0be19ab8dc011668cc50b"
                )),
                c1: Bn254Fp(hex!(
                    "9f230c739dede35fe5967f73089e4aa4041dd20ceff6b0fe120a91e199e9d523"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "431b26767084deeba5847c969880d62e693f4d3bfa99167105092c954490c413"
                )),
                c1: Bn254Fp(hex!(
                    "992428841304251f21800220eada2d3e3d63482a28b2b19f0bddb1596a36db16"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0fc20a425e476412d4b026958595fa2c301fc659afc02f07dc3c1da4b3ca5707"
                )),
                c1: Bn254Fp(hex!(
                    "9c5b4a4ce34558e8933c5771fd7d0ba26c60e2a49bb7e918b6351e3835b0a60c"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "e4a9ad1dee13e9623a1fb7b0d41416f7cad90978b8829569513f94bbd474be28"
                )),
                c1: Bn254Fp(hex!(
                    "c7aac7c9ce0baeed8d06f6c3b40ef4547a4701bebc6ab8c2997b74cbe08aa814"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
                )),
                c1: Bn254Fp(hex!(
                    "e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "7f65920905da7ba94f722c3454fb1ade89f5b67107a49d1d7d6a826aae72e91e"
                )),
                c1: Bn254Fp(hex!(
                    "c955c2707ee32157d136854130643254247725bbcd13b5d251abd4f86f54de10"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "14b26e8b5fbc3bbdd268d240fd3a7aec74ff17979863dc87bb82b2455dce4012"
                )),
                c1: Bn254Fp(hex!(
                    "4ef81b16254b5efa605574b8500fad8dbfc3d562e1ff31fd95d6b4e29f432e04"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "d718b3fb3b56156616a9423f894c2f3bfdcc9a0ad9a596cf49f8cbb85697df1d"
                )),
                c1: Bn254Fp(hex!(
                    "9b9a8957b79bc371a70283d919d80723cf4c6c6fb8c81d1243b8362c7fb7fa0b"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "3d556f175795e3990c33c3c210c38cb743b159f53cec0b4cf711794f9847b32f"
                )),
                c1: Bn254Fp(hex!(
                    "a2cb0f641cd56516ce9d7c0b1d2aae3294075ad78bcca44b20aeeb6150e5c916"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ede9dc66d08acc5ff470a8bea389d6bba35e9eca1d7ff1db4caa96986d5b272a"
                )),
                c1: Bn254Fp(hex!(
                    "644c59b2b30c4db9ba6ecfd8c7ec007632e907950e904bb18f9bf034b611a428"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "62a71e92551f8a8472ec94bef76533d3841e185ab7c0f38001a8ee645e4fb505"
                )),
                c1: Bn254Fp(hex!(
                    "26812bcd11473bc163c7de1bead28536921c0b3bb0803a9fee8afde7db5e142c"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "1894c5ed05c47d0dbaaec712f624255569184cdd540f16cfdf19b8918b8ce02e"
                )),
                c1: Bn254Fp(hex!(
                    "fcd0706a28d35917cd9fe300f89e00e7dfb80eba6f93d015b499346aa85bb71d"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "c856a8b9dd0eb15342f81baa03b7340ecdadd4b029e566c86dbbae14a3cc8716"
                )),
                c1: Bn254Fp(hex!(
                    "463cbce3eae18bc5a0909dd0adc47d18c0440b4cd35684b1b6224ad5bc55b82f"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "6dfbdc7be86e747bd342695d3dfd5f80ac259f95771cffba0aef55b778e05608"
                )),
                c1: Bn254Fp(hex!(
                    "de86a5aa2bab0c383126ff98bf31df0f4f0926ec6d0ef3a96f76d1b341def104"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
                )),
                c1: Bn254Fp(hex!(
                    "e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "66f0cb3cbc921a0ecb6bb075450933e64e44b2b5f7e0be19ab8dc011668cc50b"
                )),
                c1: Bn254Fp(hex!(
                    "9f230c739dede35fe5967f73089e4aa4041dd20ceff6b0fe120a91e199e9d523"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "04e25662a6074250e745f5d1f8e9aa68f4183446bcab39472497054c2ebe9f1c"
                )),
                c1: Bn254Fp(hex!(
                    "aed854540388fb1c6c4a6f48a78f535920f538578e939e181ec37f8708188919"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ffffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "49fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "383b7296b844bc29b9194bd30bd5866a2d39bb27078520b14d63143dbf830c29"
                )),
                c1: Bn254Fp(hex!(
                    "aba1328c3346c853f98d1af793ec75f5f0f79edc1a8e669f736a13a93d9ebd23"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "e4a9ad1dee13e9623a1fb7b0d41416f7cad90978b8829569513f94bbd474be28"
                )),
                c1: Bn254Fp(hex!(
                    "c7aac7c9ce0baeed8d06f6c3b40ef4547a4701bebc6ab8c2997b74cbe08aa814"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ede9dc66d08acc5ff470a8bea389d6bba35e9eca1d7ff1db4caa96986d5b272a"
                )),
                c1: Bn254Fp(hex!(
                    "644c59b2b30c4db9ba6ecfd8c7ec007632e907950e904bb18f9bf034b611a428"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "7f65920905da7ba94f722c3454fb1ade89f5b67107a49d1d7d6a826aae72e91e"
                )),
                c1: Bn254Fp(hex!(
                    "c955c2707ee32157d136854130643254247725bbcd13b5d251abd4f86f54de10"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "334b0e4db7cfe47eba619f27942f07abe85869ea1de273306e1d7f9b1580231e"
                )),
                c1: Bn254Fp(hex!(
                    "f90461c2f140c2412c75fdaf405bd4099e94ab1ed5451ebb93c97cfed20a362c"
                )),
            },
        ],
    ];
}
