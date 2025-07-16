#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_ecc_guest::{ed25519::Ed25519Point, eddsa::VerifyingKey};

openvm::entry!(main);

openvm::init!("openvm_init_ed25519_ed25519.rs");

pub struct Ed25519TestData {
    pub msg: &'static [u8],
    pub signature: [u8; 64],
    pub vk: [u8; 32],
}

// Test data for the non-prehash variant of Ed25519.
// The first five tests were taken from https://datatracker.ietf.org/doc/html/rfc8032#section-7.1
// The rest were randomly generated.
const ED25519_TEST_DATA: [Ed25519TestData; 10] = [
    Ed25519TestData {
        msg: b"",
        signature: hex!("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        vk: hex!("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    },
    Ed25519TestData {
        msg: &hex!("72"),
        signature: hex!("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
        vk: hex!("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    },
    Ed25519TestData {
        msg: &hex!("af82"),
        signature: hex!("6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
        vk: hex!("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"),
    },
    Ed25519TestData {
        msg: &hex!("08b8b2b733424243760fe426a4b54908632110a66c2f6591eabd3345e3e4eb98fa6e264bf09efe12ee50f8f54e9f77b1e355f6c50544e23fb1433ddf73be84d879de7c0046dc4996d9e773f4bc9efe5738829adb26c81b37c93a1b270b20329d658675fc6ea534e0810a4432826bf58c941efb65d57a338bbd2e26640f89ffbc1a858efcb8550ee3a5e1998bd177e93a7363c344fe6b199ee5d02e82d522c4feba15452f80288a821a579116ec6dad2b3b310da903401aa62100ab5d1a36553e06203b33890cc9b832f79ef80560ccb9a39ce767967ed628c6ad573cb116dbefefd75499da96bd68a8a97b928a8bbc103b6621fcde2beca1231d206be6cd9ec7aff6f6c94fcd7204ed3455c68c83f4a41da4af2b74ef5c53f1d8ac70bdcb7ed185ce81bd84359d44254d95629e9855a94a7c1958d1f8ada5d0532ed8a5aa3fb2d17ba70eb6248e594e1a2297acbbb39d502f1a8c6eb6f1ce22b3de1a1f40cc24554119a831a9aad6079cad88425de6bde1a9187ebb6092cf67bf2b13fd65f27088d78b7e883c8759d2c4f5c65adb7553878ad575f9fad878e80a0c9ba63bcbcc2732e69485bbc9c90bfbd62481d9089beccf80cfe2df16a2cf65bd92dd597b0707e0917af48bbb75fed413d238f5555a7a569d80c3414a8d0859dc65a46128bab27af87a71314f318c782b23ebfe808b82b0ce26401d2e22f04d83d1255dc51addd3b75a2b1ae0784504df543af8969be3ea7082ff7fc9888c144da2af58429ec96031dbcad3dad9af0dcbaaaf268cb8fcffead94f3c7ca495e056a9b47acdb751fb73e666c6c655ade8297297d07ad1ba5e43f1bca32301651339e22904cc8c42f58c30c04aafdb038dda0847dd988dcda6f3bfd15c4b4c4525004aa06eeff8ca61783aacec57fb3d1f92b0fe2fd1a85f6724517b65e614ad6808d6f6ee34dff7310fdc82aebfd904b01e1dc54b2927094b2db68d6f903b68401adebf5a7e08d78ff4ef5d63653a65040cf9bfd4aca7984a74d37145986780fc0b16ac451649de6188a7dbdf191f64b5fc5e2ab47b57f7f7276cd419c17a3ca8e1b939ae49e488acba6b965610b5480109c8b17b80e1b7b750dfc7598d5d5011fd2dcc5600a32ef5b52a1ecc820e308aa342721aac0943bf6686b64b2579376504ccc493d97e6aed3fb0f9cd71a43dd497f01f17c0e2cb3797aa2a2f256656168e6c496afc5fb93246f6b1116398a346f1a641f3b041e989f7914f90cc2c7fff357876e506b50d334ba77c225bc307ba537152f3f1610e4eafe595f6d9d90d11faa933a15ef1369546868a7f3a45a96768d40fd9d03412c091c6315cf4fde7cb68606937380db2eaaa707b4c4185c32eddcdd306705e4dc1ffc872eeee475a64dfac86aba41c0618983f8741c5ef68d3a101e8a3b8cac60c905c15fc910840b94c00a0b9d0"),
        signature: hex!("0aab4c900501b3e24d7cdf4663326a3a87df5e4843b2cbdb67cbf6e460fec350aa5371b1508f9f4528ecea23c436d94b5e8fcd4f681e30a6ac00a9704a188a03"),
        vk: hex!("278117fc144c72340f67d0f2316e8386ceffbf2b2428c9c51fef7c597f1d426e"),
    },
    Ed25519TestData {
        msg: &hex!("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"),
        signature: hex!("dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704"),
        vk: hex!("ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf"),
    },
    Ed25519TestData {
        msg: &hex!("470d6c430959"),
        signature: hex!("17ba04a7351648d316c9567cee48bfb568499ee0fea83fd246c44202e9ad9e920d983306ed7a3ac8ea51ebb5a1e57a0b270ca962c812aa8a89e60ce787ac8205"),
        vk: hex!("758751992ea75a6736661ac6f6ed4de7e5ed9dfbe33eaa9325780923614341d3"),
    },
    Ed25519TestData {
        msg: &hex!("a7bf65eac2bbbcb776761f247c3ccd6971396a88b3eb0bbebea592fd68b20a4d9e7cf474bea1eff3a332c9cdecd8fff2fe1e3cc6a3844318c2bc6f78a04a853ed1c535fe5824"),
        signature: hex!("a67dd824237e219fb224da3d16bcf5142b5d642e5a62198f3d2ed901eae5bb96e14975fcaeb714516fd0ded27a9fab1bada235d44d65457a96085f3d4eb0230c"),
        vk: hex!("7bfb9c93c50bb636b9d916fe3aec6a5ac6e19c47278ea404f7ea1721e3c46ced"),
    },
    Ed25519TestData {
        msg: &hex!("c781d2fd5640dd66f50c57cb7015feecf44a297c93ceafb611acffb39cb7c69f277491cc39eaa836008194e77860a7716799eca708859188b46d3e44dd3f57f3553244b1a8e5092fe1bdd6e016b67fd94e88187d03efe25d4178266dcac56aa1"),
        signature: hex!("63afcc5c9b282e2e7e8871b411cd69e1cad83f057cb764862453af88ed5bb255ebf96dab5ea1b1041bdc6d515e79f4c774e9c87d7b7a681cf399cab3005a580e"),
        vk: hex!("7e2467b9b1ae68d0b79e9c8214592022be6c369b2ba771cd7100d4be0db554b3"),
    },
    Ed25519TestData {
        msg: &hex!("7d89777ab2b5ff2ae46da312ed32a48b22977eb52a11fc3ab355f3ad7ad40641218681eef5add98f01"),
        signature: hex!("cf76c790c77166e2db3a28eca5de7f42ffc9dac85895de1f929c72714d23a9fdb92017432ef7424ff14acb815c76881d55dcc80cca1ed8630473baa9b9b9d005"),
        vk: hex!("4ff316d580e2330d99f92fdc149d4c88f36981be132a4f3fa065e649cf3571a8"),
    },
    Ed25519TestData {
        msg: &hex!("3d98781c525e466626b418"),
        signature: hex!("185a5fd9b6e82e07c72bc81296cb1e4f7a5bfd4f5226961f52e24c0f20cf12310b740d38146dfdba662be9b6b2926712a648e73fe22239486149a404864df50f"),
        vk: hex!("99cd13488f1b48f6d57d1f77ce2006487e65f8e7d6f1936cf6f36adf0b602d55"),
    },
];

// Test data for the prehash variant of Ed25519.
// First test was taken from the RFC: https://datatracker.ietf.org/doc/html/rfc8032#section-7.3
// The rest were randomly generated.
const ED25519PH_TEST_DATA: [Ed25519TestData; 10] = [
    // This test is taken from the RFC: https://datatracker.ietf.org/doc/html/rfc8032#section-7.3
    Ed25519TestData {
        msg: b"abc",
        signature: hex!(
            "98a70222f0b8121aa9d30f813d683f809e462b469c7ff87639499bb94e6dae4131f85042463c2a355a2003d062adf5aaa10b8c61e636062aaad11c2a26083406"
        ),
        vk: hex!(
            "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf"
        ),
    },
    Ed25519TestData {
        msg: &hex!("4023e5edbfde97998cea65ee971c8cb24526596044f1216fa2c0d8c8ec8df95ef237bdd314022a2780dd09b9dcb8ba1df76d6ac7f0d7bf4374ef6405979dad73490d2b363545a2c0f5eddb965705a565f44a371d5cf58004d6834e0271c5e674"),
        signature: hex!("5ac8ece6e00341bb1bc10403837f2f59fc0a3cbcd352e9101dccb5af2ad41da9199758ad606679bd2dc4af5a1d89c73c36365ae5455b725c6a2cea8d06399501"),
        vk: hex!("6c6193110409068064bc7986acbc3c96449dfe32891e6c2fb3fc33ad7655ba0b")
    },
    Ed25519TestData {
        msg: &hex!("65b5f8e0fb2f47a16bc0e6777b5a4abed8a8dbcbd0b685257e47ede83a433cc3c5d8755959cdf8caa6990eae48f3759b03593b9bc0d8fc7383a5d8ea7b02de9dd761b410a4"),
        signature: hex!("93406bdd8d40925e1a6e316654874444baca364f6700c074e2975e50cd2e708e17d10084574701fe0c91eda4d2d796e26b2aba67c48b3f94fac151e699cfe809"),
        vk: hex!("9b606b002b395f9efe41a3f388d37bb81ed52486f8ddc19996176462da5d7b29")
    },
    Ed25519TestData {
        msg: &hex!("62dba7573793c7fa9908c4feb690c0b61b136d5c744b69b343a61bdc"),
        signature: hex!("c4bd72c7935dc4d3784abfcb7f20124429b73925d01ec48673bd37ab26c2cc159089ade51df1eb7ce175abf43afd6c23c7bb39ad2d6476acb3a04ce7339f2e0f"),
        vk: hex!("a166560aa0d208bf06c93a2a7f64748e503def9407ca8ee81687d7e6ee21efa7")
    },
    Ed25519TestData {
        msg: &hex!("40033e866996ddca4ff7a48d557bc7a4ffc4d97274bfae4691976cf0d587a9d5823a38ed5e7314b67b61b5d7536d3d581bd0ce77ca27ebd2ce26ce2e"),
        signature: hex!("24ea129ba9abc5b11dc5a690bccbaebe315b8882b029b8c81bd8cc6a5b65e79aa82298b64fc61e03d4081642c90a60ad3955ab484304194d95ecf1ba8dfe760a"),
        vk: hex!("bed62265ab0d6ba1be8dfe009fbb9514c6774ead6c34492adaee4893c32d39fc")
    },
    Ed25519TestData {
        msg: &hex!("0e9e0304a804628305083e4bde6de5d82fe5f5dbb47e232f2cf14439fd36dd59f26b87574614d8f4af6019c5d4d7ec77fee102445faf0c75f635a31234e2135199df5cdf013ff3472346e6f69e8a"),
        signature: hex!("dcf2a479abd7ea5211d853279f128e402fb64fc3148780694422e8a572e29ed1557fd89f172c0a3c1b2ab6944297deace095583bff09b302936f64198385490e"),
        vk: hex!("47547e699eaf210dcce343b4a2d176607a8a1ceb2a3e912360f40dd1fa3ab216")
    },
    Ed25519TestData {
        msg: &hex!("563df2cb74d0cfd961ed010958845e6983b1ca7a55761dba35ccebcf17dfd972bfcb908c116a4eacb84235ed"),
        signature: hex!("9960cb5936c23969707bf92ab0d51ae941dc2ce2534d818ed1c829dbbe916a93657bc6c7d38ad5f1d07df513f35409d9581cb3c122f0742f41295dc8e0396d0c"),
        vk: hex!("4c6a352b76f20b7757ee96083f2cd8b759556551d9e5937a7ec4345d4bf974ec")
    },
    Ed25519TestData {
        msg: &hex!("de08bd2b3a014ea5a85de35f718fea45e6c23c06f2d23e5bbc22bb998de9d7cb9d21fbb9a55aff4d2a867daaf4a897281c889c9536fb2259014030d8b24d9a04dfc0b74e62847638f1740767e62ccfbfab174749f657a75a3845924ca6a6a6d539"),
        signature: hex!("64bbb532b1ebe08554d6569be1133ed9219e0450fd1bf12fd30ce6aa3150aea3054ed3ff35ee7c458eeeee6141a9e5f3bb4d14c345a4435ae331ed3ab0fa350e"),
        vk: hex!("e22e10095f6fb59e5714093b78f9889ac3aa394c2a87b858ca6afb8558241a2e")
    },
    Ed25519TestData {
        msg: &hex!("96d3dd0d9df5160827ad62c96dc04ff1621e50b3c7a2bcba53e6e2c83b6939"),
        signature: hex!("5f6d536c7cb27391f7a274fd7f72e258b49f511476e5a0cd562c3f52a050694734c7251434bde23fe14977094a482fe5ff14a237fc24804cab5e595916240401"),
        vk: hex!("83a53511df806437a5c3e72c1fd43d448e217da4cc789d817fa2b12d9628128b")
    },
    Ed25519TestData {
        msg: &hex!("c366ea5e7b41a0382eaa22a5d6101d9fef9a7b234cc54f218108f2695f288c9369723aa7958d0605166b52be9a895133ae77eba9122d8780dd897d3f7c871cc48f57c11fcfa7a5a056c1e1"),
        signature: hex!("535e3a9fc9a3948998398f8d79af40ff7cb320e62498df3db3ad7cfbbd5c02eedee95a11f745f5334f6532e36ec5cfca1c2e14e416481e3cc3600ef58dab4901"),
        vk: hex!("4028f8f8d4f461fcc41d2c2878eddb7acb9fbf3254e7de511c3f513673ff05d0")
    },
];

pub fn main() {
    for test_data in ED25519_TEST_DATA {
        let vk = VerifyingKey::<Ed25519Point>::from_bytes(&test_data.vk).unwrap();
        assert!(vk.verify(test_data.msg, &test_data.signature).is_ok());
    }

    for test_data in ED25519PH_TEST_DATA {
        let vk = VerifyingKey::<Ed25519Point>::from_bytes(&test_data.vk).unwrap();
        assert!(vk
            .verify_ph(test_data.msg, None, &test_data.signature)
            .is_ok());
    }
}
