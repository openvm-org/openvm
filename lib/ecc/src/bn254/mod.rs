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

pub struct Bn254;

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
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "70e4c9dcda350bd676212f29081e525c608be676dd9fb9e8dfa765281cb78412"
                )),
                c1: Bn254Fp(hex!(
                    "
        ac62f3805ff05ccae5c7ee8e779279748e0b1512fe7c32a6e6e7fab4f3966924"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "3d556f175795e3990c33c3c210c38cb743b159f53cec0b4cf711794f9847b32f"
                )),
                c1: Bn254Fp(hex!(
                    "
        a2cb0f641cd56516ce9d7c0b1d2aae3294075ad78bcca44b20aeeb6150e5c916"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
                )),
                c1: Bn254Fp(hex!(
                    "
        e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "62a71e92551f8a8472ec94bef76533d3841e185ab7c0f38001a8ee645e4fb505"
                )),
                c1: Bn254Fp(hex!(
                    "
        26812bcd11473bc163c7de1bead28536921c0b3bb0803a9fee8afde7db5e142c"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "2f69b7ea10c8a22ed31baa559b455c42f43f35a461363ae94986794fe7c18301"
                )),
                c1: Bn254Fp(hex!(
                    "
        4b2c0c6eeeb8c624c02a8e6799cb80b07d9f72c746b27fa27506fd76caf2ac12"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "49fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ffffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "7fa6d41e397d6fe84ad255be8db34c8990aaacd08c60e9efbbe482cccf81dc19"
                )),
                c1: Bn254Fp(hex!(
                    "
        01c1c0f42baa9476ec39d497e3a5037f9d137635e3eecb06737de70bb6f8ab00"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "6dfbdc7be86e747bd342695d3dfd5f80ac259f95771cffba0aef55b778e05608"
                )),
                c1: Bn254Fp(hex!(
                    "
        de86a5aa2bab0c383126ff98bf31df0f4f0926ec6d0ef3a96f76d1b341def104"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ede9dc66d08acc5ff470a8bea389d6bba35e9eca1d7ff1db4caa96986d5b272a"
                )),
                c1: Bn254Fp(hex!(
                    "
        644c59b2b30c4db9ba6ecfd8c7ec007632e907950e904bb18f9bf034b611a428"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "66f0cb3cbc921a0ecb6bb075450933e64e44b2b5f7e0be19ab8dc011668cc50b"
                )),
                c1: Bn254Fp(hex!(
                    "
        9f230c739dede35fe5967f73089e4aa4041dd20ceff6b0fe120a91e199e9d523"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "431b26767084deeba5847c969880d62e693f4d3bfa99167105092c954490c413"
                )),
                c1: Bn254Fp(hex!(
                    "
        992428841304251f21800220eada2d3e3d63482a28b2b19f0bddb1596a36db16"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0fc20a425e476412d4b026958595fa2c301fc659afc02f07dc3c1da4b3ca5707"
                )),
                c1: Bn254Fp(hex!(
                    "
        9c5b4a4ce34558e8933c5771fd7d0ba26c60e2a49bb7e918b6351e3835b0a60c"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "e4a9ad1dee13e9623a1fb7b0d41416f7cad90978b8829569513f94bbd474be28"
                )),
                c1: Bn254Fp(hex!(
                    "
        c7aac7c9ce0baeed8d06f6c3b40ef4547a4701bebc6ab8c2997b74cbe08aa814"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
                )),
                c1: Bn254Fp(hex!(
                    "
        e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "7f65920905da7ba94f722c3454fb1ade89f5b67107a49d1d7d6a826aae72e91e"
                )),
                c1: Bn254Fp(hex!(
                    "
        c955c2707ee32157d136854130643254247725bbcd13b5d251abd4f86f54de10"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "14b26e8b5fbc3bbdd268d240fd3a7aec74ff17979863dc87bb82b2455dce4012"
                )),
                c1: Bn254Fp(hex!(
                    "
        4ef81b16254b5efa605574b8500fad8dbfc3d562e1ff31fd95d6b4e29f432e04"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "d718b3fb3b56156616a9423f894c2f3bfdcc9a0ad9a596cf49f8cbb85697df1d"
                )),
                c1: Bn254Fp(hex!(
                    "
        9b9a8957b79bc371a70283d919d80723cf4c6c6fb8c81d1243b8362c7fb7fa0b"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "3d556f175795e3990c33c3c210c38cb743b159f53cec0b4cf711794f9847b32f"
                )),
                c1: Bn254Fp(hex!(
                    "
        a2cb0f641cd56516ce9d7c0b1d2aae3294075ad78bcca44b20aeeb6150e5c916"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ede9dc66d08acc5ff470a8bea389d6bba35e9eca1d7ff1db4caa96986d5b272a"
                )),
                c1: Bn254Fp(hex!(
                    "
        644c59b2b30c4db9ba6ecfd8c7ec007632e907950e904bb18f9bf034b611a428"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "62a71e92551f8a8472ec94bef76533d3841e185ab7c0f38001a8ee645e4fb505"
                )),
                c1: Bn254Fp(hex!(
                    "
        26812bcd11473bc163c7de1bead28536921c0b3bb0803a9fee8afde7db5e142c"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "1894c5ed05c47d0dbaaec712f624255569184cdd540f16cfdf19b8918b8ce02e"
                )),
                c1: Bn254Fp(hex!(
                    "
        fcd0706a28d35917cd9fe300f89e00e7dfb80eba6f93d015b499346aa85bb71d"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "0100000000000000000000000000000000000000000000000000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "c856a8b9dd0eb15342f81baa03b7340ecdadd4b029e566c86dbbae14a3cc8716"
                )),
                c1: Bn254Fp(hex!(
                    "
        463cbce3eae18bc5a0909dd0adc47d18c0440b4cd35684b1b6224ad5bc55b82f"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "6dfbdc7be86e747bd342695d3dfd5f80ac259f95771cffba0aef55b778e05608"
                )),
                c1: Bn254Fp(hex!(
                    "
        de86a5aa2bab0c383126ff98bf31df0f4f0926ec6d0ef3a96f76d1b341def104"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "5a13a071460154dc9859c9a9ede0aadbb9f9e2b698c65edcdcf59a4805f33c06"
                )),
                c1: Bn254Fp(hex!(
                    "
        e3b02326637fd382d25ba28fc97d80212b6f79eca7b504079a0441acbc3cc007"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "66f0cb3cbc921a0ecb6bb075450933e64e44b2b5f7e0be19ab8dc011668cc50b"
                )),
                c1: Bn254Fp(hex!(
                    "
        9f230c739dede35fe5967f73089e4aa4041dd20ceff6b0fe120a91e199e9d523"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "04e25662a6074250e745f5d1f8e9aa68f4183446bcab39472497054c2ebe9f1c"
                )),
                c1: Bn254Fp(hex!(
                    "
        aed854540388fb1c6c4a6f48a78f535920f538578e939e181ec37f8708188919"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ffffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "feffff77314763574f5cdbacf163f2d4ac8bd4a0ce6be2590000000000000000"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "48fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "49fd7c60e544bde43d6e96bb9f068fc2b0ccace0e7d96d5e29a031e1724e6430"
                )),
                c1: Bn254Fp(hex!(
                    "
        0000000000000000000000000000000000000000000000000000000000000000"
                )),
            },
        ],
        [
            Fp2 {
                c0: Bn254Fp(hex!(
                    "383b7296b844bc29b9194bd30bd5866a2d39bb27078520b14d63143dbf830c29"
                )),
                c1: Bn254Fp(hex!(
                    "
        aba1328c3346c853f98d1af793ec75f5f0f79edc1a8e669f736a13a93d9ebd23"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "e4a9ad1dee13e9623a1fb7b0d41416f7cad90978b8829569513f94bbd474be28"
                )),
                c1: Bn254Fp(hex!(
                    "
        c7aac7c9ce0baeed8d06f6c3b40ef4547a4701bebc6ab8c2997b74cbe08aa814"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "ede9dc66d08acc5ff470a8bea389d6bba35e9eca1d7ff1db4caa96986d5b272a"
                )),
                c1: Bn254Fp(hex!(
                    "
        644c59b2b30c4db9ba6ecfd8c7ec007632e907950e904bb18f9bf034b611a428"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "7f65920905da7ba94f722c3454fb1ade89f5b67107a49d1d7d6a826aae72e91e"
                )),
                c1: Bn254Fp(hex!(
                    "
        c955c2707ee32157d136854130643254247725bbcd13b5d251abd4f86f54de10"
                )),
            },
            Fp2 {
                c0: Bn254Fp(hex!(
                    "334b0e4db7cfe47eba619f27942f07abe85869ea1de273306e1d7f9b1580231e"
                )),
                c1: Bn254Fp(hex!(
                    "
        f90461c2f140c2412c75fdaf405bd4099e94ab1ed5451ebb93c97cfed20a362c"
                )),
            },
        ],
    ];
}
