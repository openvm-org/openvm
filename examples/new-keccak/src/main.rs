// [!region imports]
// [!endregion imports]

// [!region main]
use hex_literal::hex;
use openvm as _;
use revm_primitives::keccak256;

/// Vector of test cases for Keccak-256 hash function.
/// Each test case consists of (input_bytes, expected_hash_result).
const KECCAK_TEST_CASES: &[(&[u8], [u8; 32])] = &[
    (
        b"",
        hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    ),
    (
        b"hello world",
        hex!("47173285a8d7341e5e972fc677286384f802f8ef42a5ec5f03bbfa254cb01fad"),
    ),
    (
        &hex!("91376f5774419e518ced240cfb7ce0fbb835d80c609f7db3736a8474381c23cb7dde504e009f8d9656351a22ff18172046cbd80e074ee99ac7cc95f0daf32c35d4a0b41932d6efee42790e0298fa3d8f0d362f79008e9bd9989b355470e714aebd0d5936bbbfcc11d4"),
        hex!("bcd603061b992b3c881b2418a691c4a77606f4e9ff4fb1e57bd64118430f9e64"),
    ),
    (
        &hex!("2e8b467f65bb7a367e17b0f1ea54e20cfcd5f581bb7a9e937b1d48e289cb8477a4aa081031666529e3659782945bdadc3524e069b1e50da71caa04e681e986869325ba36e2fd"),
        hex!("f5856263b398ac1b9682363fd87f3f47d96c213d9a27ca84f42ccdc69c04b600"),
    ),
    (
        &hex!("8c7e1ae8d32e176dbb4c25dfbfa4edf2c642"),
        hex!("1d673221b76ffb970ef7abec781e7e8678ef7535b95ffc9dd7e7b9ebb472e155"),
    ),
    (
        &hex!("1f483a587db0a0152083be135d4418a3387a9d223a3329b8044ac6b97dfb754718cec558ab6cf6fb0f46b344442326b7d639ec7b81e0eac95340bb3e96860368961a82d98f35eb893dd848da52e7a7cc3d970657592071da1ea70ce99c995d2bdf4f9f764f2ca277705843f0800e7a97"),
        hex!("b83494263c50b8444ae444d436ed9a30065ce869342d9c3a5fe319ed323b3ac6"),
    ),
    (
        &hex!("a2d386e126c7"),
        hex!("d94338cbf5a5505c6079f381bc9c5f895a93911ea906d84ccf1c876ae24f31f9"),
    ),
    (
        &hex!("590307764a0f2294f82301735054adc163cd1ee5c582fc3d173f54ba7c5f6dbe72aa30ce8007657d7bafb9d083e80574b3d3a69469ed43cf61cdd8bb2d2762a0861affae378c052dad637707f420cfbc1dbbf7"),
        hex!("03c2229015d2693d8448b1761c04ae80cee6e11e046ae2d3ebeb4291531abf8d"),
    ),
    (
        &hex!("8b0d83e9b5d1770faa1f17195a9c8fa74bd5109e50ba74f752d1a40189b0f2d7dd8abe3b0c9648f67f413f584ccea0f69fa2346460dd97a28acb1d0ba8618398817c76be800cebce0d701595ac0b5f422775c504"),
        hex!("3390695701398af69128d9d8fc4a8a1c43c340e65b673a896dd40360c4c816f9"),
    ),
    (
        &hex!("2f80c45d979eb3f7"),
        hex!("12292d148143f86721726f656ae496b95d113dd669f78e202ac2cca9bca8304c"),
    ),
    (
        &hex!("d43a4df4bf208321971cabff80d9ae2bf619bfac98004b729e6415a481bf70089a39ad5019f21d4bec3dd8f2598cf9fccc6a5fcd5ac79d516d1fdbb123b4bda96021130b44fde0633c50ae85c376db29b551edb16d5fac451b0cbea06356f80661b011f3904e3c62ea9f8ad3773da5"),
        hex!("1cf0c4b5afb89f146d9b09d878d4afc00aa8a56037ff8602071ce989f3e58695"),
    ),
    (
        &hex!("72ba3368324d809df9a9d97e251e00a6dfb3e70fca2892e63ee3df6d5b08e01e63c9db23296586bf9a3ee78a68"),
        hex!("7d79f8ed4fb4cd728d9ca383536f81f025d8e246504f5e9d53cbfa05006abad1"),
    ),
    (
        &hex!("6b0bd4f931c1d8861f5ccea87642791dfa74fe8544175fb63451ee0adcfcc13f3040ab8f7525892fcd7da15ce23da3b2f21a6e55dca0765daeb2407acff87c03b7d1a13d3acd77390d6055fde3eeeeef381c04cd41d5c83bd3ef7e494a8391de3a6d8f1f1941e9fb217bd8"),
        hex!("9550d5094b85b0d1f01209b71c338114a09aad20cb33b7d444c31c1e1473716b"),
    ),
    (
        &hex!("494c7a2642a3d6b3943605cbc30140aaf622f3a34338fdff183e5ef6bc41389d6f095056760dd144f0a14a2b21ca5b487cab0a3ea23a86"),
        hex!("a03c900c3ebf6c4037f8351b1a144720a04a105e067707c5cbac79aade25fb56"),
    ),
    (
        &hex!("4d91a527e6bd5ea378c5ba9516b8b34cc99dd7c52ed03b54623f1ce4ca6d886b37c95dde605fb7ab8274f21488229bd9169e760703747ae9d23699f77db579f1ad31ef8dbdb57cce0de326db8bbc6550a5cc5fe09c8486ef24ef023b4f2f8aa32e390ede54eb3d6123fbc08a8e591d6cad80c3fe52821da10136cce9"),
        hex!("5b47a38b57bfe74950e59da3c1918d94293d4250b8991a3324ec42160b3d402c"),
    ),
    (
        &hex!("9e60df2407f3057680e332f32cafb32e195aa13c275c1571c4877da2f26b00ed139f808310e9e4a800fd950ddbb3dc"),
        hex!("52474f4d926fa0599845e38ab4a9529737bd42f0d46784d5f584c7ab063c18f7"),
    ),
    (
        &hex!("712b9a6afe96f461e31092ffe38d74288cb1424e876a3592bca03dbc9c3a57"),
        hex!("3b0a1d9106e06e1a3de754fc90583862e82ec92f81edf261e5a12108d46a69bf"),
    ),
    (
        &hex!("9feff865dad6dc06e095d330f363178b71e56fd1b86fcac2249aed2a84a82fa4d596d6f4142b71ec034f543d69bdbc47404b8146203b215ce3bce1a8c4"),
        hex!("63fe4b8bd1e0d823b5d7b6ab722c5b5dfcfe472da792e8974a38cc81fdd53777"),
    ),
    (
        &hex!("0213643f8bbb0aa1e9528f7f117c050cbd9d8709cc3672850363c0ddef5be2001c"),
        hex!("4bf5a88bb8954c539620033214e1b1e34306639417c7b7683a4aa825f5c004bb"),
    ),
    (
        &hex!("a79eca9f55b0f13c533ca91e71495f65f642ab91224bbe664ff59b1b3362ba51"),
        hex!("a710b4cbdc60423e7ab1773d6731ceb3ea0157cff6df39a1b8c1410acf8c0075"),
    ),
    (
        &hex!("4cc8abc63b48a1a653c3a52801394db7ae608249d67006329fd259ec203a1ed836f384ed88b6dff310948c9f29f7c240fc49d705299e8aec3239e35c998c2befbd8997e59c8581"),
        hex!("ba972e77578147700ba7539b89e53fc40bdd9893b3cccef0bcf1a2f7ef65c4bc"),
    ),
    (
        &hex!("f66cd0cd9a55d6c2046dd7df0cf6496a876d165d0de8f639140cafa2155abc2426cc55258e0db47938d262472558f4eb63e98cfd3725296dc1de7c57d9d148194c2ac3b9d0c267814578bfc66c46da093bc59f132b41076865659d548ed3"),
        hex!("54ea328ab976e794e653b82d5cfbc0f20305e6e33a97bf4e8764d06bbed2753c"),
    ),
    (
        &hex!("fe7eee2bcc4071ab2bcd801d805cc082bc0d7781713941e06e4819e09748a5692c945f3b3b0b632b50bce1"),
        hex!("57a655827352ff95ce21082515add7f6fb0f51b457e338a1cfd7a5912a2c30a0"),
    ),
    (
        &hex!("2cf005de4865b6321e8995f38825353e64e34f7ecaedd9b5f39ee41067e9e63f38618985d015b1fb89be1140e249ac92d0b8f273"),
        hex!("0ba80a1c481facdec27552c2d8382467fb97f864a6c915e2882045a995dd452a"),
    ),
    (
        &hex!("edee326b959740e515dd8a0ed9ced665d782b50bda2af634189def905b59d3a9ac635e1d96c670eaf28675b0071db0d86131088fad00a0b80da10dd1259692d17dc9af43a5e3f40185e88776be68bdf2a3ddeb7c9726e5bde8e4d14bcf6e991d142047"),
        hex!("295d1a5a86a4ea5cc3d7c5860c3fa6069374a610e5f9cda1e643b4f01e957752"),
    ),
    (
        &hex!("8703f576434a9ec235904e106991aa447b1d4e715fd1fe2e5aa2e3aa950291d624d97420dc9eec10d76cdf23ae189c995760f339c20bbb5e"),
        hex!("7ebb76c5e9669e9bceb867560682a653e600e116af83777ed6e71fb4b28f1b48"),
    ),
    (
        &hex!("e8dffb080d61e60561a5f2e44db8e65c4ab134f3283411b374da96cc6a94408871de72d5b8e39aae83c81fa7b31027a8da64915010eb688a72f5f605aab2b77d67baa1f90056281e92fa3031108e2061a1b110f4d0d62ff91f956f977cc504a68337d7b01d2ff6caadc57e"),
        hex!("e85f50155c3abf4d0cabe2a62d9af075c163600fb0c12e1bc3d004ce4d25bd13"),
    ),
    (
        &hex!("5de9ef34a6f71cbe1780ff29816f7b26f574f0ea5d6adeaff6d9c78e08440ce9c3a5597e31693e0d2a89b1b84bf3b8c8cda9c13b7abb3bf8b2cf0cb3410d9df1e6fd4765546ce8f4e1d821aa7253fdf044437a65eca5d7393716"),
        hex!("3de18f83084b4ced8b0a261f80225b52cf9c27950dff768bb2e0ab1a8f015fe2"),
    ),
    (
        &hex!("8278f74015c08ea408c53446d4e9364eddc6c37cf2a03b54df9a9122095eef5f3812d03afa4f7c52ba937f73968e3074ed5ddd7a6439916b8c4a6bb821896852bbe1430be00f9da2244d980834696f949232facdc0e214f52f308267"),
        hex!("1f60d8c4bfd1cf910fd7e5a5ecd3e2996aee20d7c40c1f6d43225d832b937b0a"),
    ),
    (
        &hex!("65063a451690b64af8901e2cc0414abbe0e92c6789c721577bf975b4cd58771086ad8f6e2899bf45e7f23e76a299accae8782dd1f47fa449d7cc7f06ced46bd7bde7ce5f61e594485a6489edd5dc96ad79c5425960297b028f9669fb827a"),
        hex!("30d3360dbb001c115a287da3eac988b1a925b2ab607ba1e1f909a64eef1c90e4"),
    ),
    (
        &hex!("41c7813c97"),
        hex!("be714f64489c82a8fc1008db96c096bd72b74ca142f8682c19aa5d7b8d575d31"),
    ),
    (
        &hex!("7a9fa65269a9af8e994849b91c58c9ad2b4da115a96514a02200d931c359e4a26ff54c3937ed061a1e58e7cf2c1b1dd72c42049081fe28dd59caaf09447c403c167265d5e121e367d41fc87c3c1c9a81c582ee826cfe"),
        hex!("41ca49571619a8b82cd7bff0e317493eb70d8dfaee994702c3c1ad80b794c863"),
    ),
    (
        &hex!("d8b70735a52277d4e361a68b884abf5118756c5685dd0af29e8274b94ab37bc126dbb8e97735cf"),
        hex!("a43a2a0321aac66ed239bc40518f9947ae4480bc70d4b1c23bbe5d1a367e1cd1"),
    ),
    (
        &hex!("190dcb4311e6c20f303e939bea29a0f42856735821743d92579d1d2e41b0925304e08be11e978aa760993083bf546a31d8e1877b405f14f3bdcae789ed5d639e51ac846ab2390ae9552e733a7233"),
        hex!("f5392ee04880a0bd1336f30ee79b5c014a90728bf29f422dabb4ae6bc972f30b"),
    ),
];
// todo: call the forked tiny keccak library once that is updated instead of directly calling the
// keccak256_guest native functions
pub fn main() {
    #[cfg(target_os = "zkvm")]
    {
        for &(input, expected) in KECCAK_TEST_CASES {
            let mut input = input.to_vec();
            let mut output = [0u8; 32];
            openvm_new_keccak256_guest::native_keccak256(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
            );
            assert_eq!(output, expected);
        }

        // let mut expected_output = [0u8; 32];
        // openvm_keccak256::keccak256(&buffer);
        // assert_eq!(output, expected_output);
    }
}
// [!endregion main]
