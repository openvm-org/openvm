use ecc_test::*;
use ecdsa::{EncodedPoint, RecoveryId, Signature, SigningKey, VerifyingKey};
use elliptic_curve::group::Curve;
use hex_literal::hex;
use openvm_ecc_guest::{algebra::IntMod, weierstrass::WeierstrassPoint, CyclicGroup};
use openvm_sha256_guest::sha256;

// Taken from https://github.com/RustCrypto/elliptic-curves/blob/32343a78f1522aa5bd856556f114053d4bb938e0/k256/src/test_vectors/group.rs#L9
pub const ADD_TEST_VECTORS: &[([u8; 32], [u8; 32])] = &[
    (
        hex!("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"),
        hex!("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"),
    ),
    (
        hex!("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5"),
        hex!("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A"),
    ),
    (
        hex!("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9"),
        hex!("388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672"),
    ),
    (
        hex!("E493DBF1C10D80F3581E4904930B1404CC6C13900EE0758474FA94ABE8C4CD13"),
        hex!("51ED993EA0D455B75642E2098EA51448D967AE33BFBDFE40CFE97BDC47739922"),
    ),
    (
        hex!("2F8BDE4D1A07209355B4A7250A5C5128E88B84BDDC619AB7CBA8D569B240EFE4"),
        hex!("D8AC222636E5E3D6D4DBA9DDA6C9C426F788271BAB0D6840DCA87D3AA6AC62D6"),
    ),
    (
        hex!("FFF97BD5755EEEA420453A14355235D382F6472F8568A18B2F057A1460297556"),
        hex!("AE12777AACFBB620F3BE96017F45C560DE80F0F6518FE4A03C870C36B075F297"),
    ),
    (
        hex!("5CBDF0646E5DB4EAA398F365F2EA7A0E3D419B7E0330E39CE92BDDEDCAC4F9BC"),
        hex!("6AEBCA40BA255960A3178D6D861A54DBA813D0B813FDE7B5A5082628087264DA"),
    ),
    (
        hex!("2F01E5E15CCA351DAFF3843FB70F3C2F0A1BDD05E5AF888A67784EF3E10A2A01"),
        hex!("5C4DA8A741539949293D082A132D13B4C2E213D6BA5B7617B5DA2CB76CBDE904"),
    ),
    (
        hex!("ACD484E2F0C7F65309AD178A9F559ABDE09796974C57E714C35F110DFC27CCBE"),
        hex!("CC338921B0A7D9FD64380971763B61E9ADD888A4375F8E0F05CC262AC64F9C37"),
    ),
    (
        hex!("A0434D9E47F3C86235477C7B1AE6AE5D3442D49B1943C2B752A68E2A47E247C7"),
        hex!("893ABA425419BC27A3B6C7E693A24C696F794C2ED877A1593CBEE53B037368D7"),
    ),
    (
        hex!("774AE7F858A9411E5EF4246B70C65AAC5649980BE5C17891BBEC17895DA008CB"),
        hex!("D984A032EB6B5E190243DD56D7B7B365372DB1E2DFF9D6A8301D74C9C953C61B"),
    ),
    (
        hex!("D01115D548E7561B15C38F004D734633687CF4419620095BC5B0F47070AFE85A"),
        hex!("A9F34FFDC815E0D7A8B64537E17BD81579238C5DD9A86D526B051B13F4062327"),
    ),
    (
        hex!("F28773C2D975288BC7D1D205C3748651B075FBC6610E58CDDEEDDF8F19405AA8"),
        hex!("0AB0902E8D880A89758212EB65CDAF473A1A06DA521FA91F29B5CB52DB03ED81"),
    ),
    (
        hex!("499FDF9E895E719CFD64E67F07D38E3226AA7B63678949E6E49B241A60E823E4"),
        hex!("CAC2F6C4B54E855190F044E4A7B3D464464279C27A3F95BCC65F40D403A13F5B"),
    ),
    (
        hex!("D7924D4F7D43EA965A465AE3095FF41131E5946F3C85F79E44ADBCF8E27E080E"),
        hex!("581E2872A86C72A683842EC228CC6DEFEA40AF2BD896D3A5C504DC9FF6A26B58"),
    ),
    (
        hex!("E60FCE93B59E9EC53011AABC21C23E97B2A31369B87A5AE9C44EE89E2A6DEC0A"),
        hex!("F7E3507399E595929DB99F34F57937101296891E44D23F0BE1F32CCE69616821"),
    ),
    (
        hex!("DEFDEA4CDB677750A420FEE807EACF21EB9898AE79B9768766E4FAA04A2D4A34"),
        hex!("4211AB0694635168E997B0EAD2A93DAECED1F4A04A95C0F6CFB199F69E56EB77"),
    ),
    (
        hex!("5601570CB47F238D2B0286DB4A990FA0F3BA28D1A319F5E7CF55C2A2444DA7CC"),
        hex!("C136C1DC0CBEB930E9E298043589351D81D8E0BC736AE2A1F5192E5E8B061D58"),
    ),
    (
        hex!("2B4EA0A797A443D293EF5CFF444F4979F06ACFEBD7E86D277475656138385B6C"),
        hex!("85E89BC037945D93B343083B5A1C86131A01F60C50269763B570C854E5C09B7A"),
    ),
    (
        hex!("4CE119C96E2FA357200B559B2F7DD5A5F02D5290AFF74B03F3E471B273211C97"),
        hex!("12BA26DCB10EC1625DA61FA10A844C676162948271D96967450288EE9233DC3A"),
    ),
];

// Taken from https://github.com/RustCrypto/elliptic-curves/blob/32343a78f1522aa5bd856556f114053d4bb938e0/k256/src/arithmetic/projective.rs#L797
#[test]
fn test_vector_repeated_add() {
    let generator = Secp256k1Point::GENERATOR;
    let mut p = generator;

    for test_vector in ADD_TEST_VECTORS {
        let affine = p.to_affine();

        let (expected_x, expected_y) = test_vector;
        assert_eq!(&affine.x().to_be_bytes(), expected_x);
        assert_eq!(&affine.y().to_be_bytes(), expected_y);

        p += &generator;
    }
}

/// Signature recovery test vectors
struct RecoveryTestVector {
    pk: [u8; 33],
    msg: &'static [u8],
    sig: [u8; 64],
    recid: RecoveryId,
}

const RECOVERY_TEST_VECTORS: &[RecoveryTestVector] = &[
    // Recovery ID 0
    RecoveryTestVector {
        pk: hex!("021a7a569e91dbf60581509c7fc946d1003b60c7dee85299538db6353538d59574"),
        msg: b"example message",
        sig: hex!(
            "ce53abb3721bafc561408ce8ff99c909f7f0b18a2f788649d6470162ab1aa032
                 3971edc523a6d6453f3fb6128d318d9db1a5ff3386feb1047d9816e780039d52"
        ),
        recid: RecoveryId::new(false, false),
    },
    // Recovery ID 1
    RecoveryTestVector {
        pk: hex!("036d6caac248af96f6afa7f904f550253a0f3ef3f5aa2fe6838a95b216691468e2"),
        msg: b"example message",
        sig: hex!(
            "46c05b6368a44b8810d79859441d819b8e7cdc8bfd371e35c53196f4bcacdb51
                 35c7facce2a97b95eacba8a586d87b7958aaf8368ab29cee481f76e871dbd9cb"
        ),
        recid: RecoveryId::new(true, false),
    },
];

#[test]
fn public_key_recovery() {
    for vector in RECOVERY_TEST_VECTORS {
        let digest = sha256(vector.msg);
        let sig: Signature<Secp256k1> = Signature::try_from(vector.sig.as_slice()).unwrap();
        let recid = vector.recid;
        let pk = VerifyingKey::recover_from_prehash(digest.as_slice(), &sig, recid).unwrap();
        assert_eq!(&vector.pk[..], EncodedPoint::from(&pk).as_bytes());
    }
}

/// End-to-end example which ensures RFC6979 is implemented in the same
/// way as other Ethereum libraries, using HMAC-DRBG-SHA-256 for RFC6979,
/// and Keccak256 for hashing the message.
///
/// Test vectors adapted from:
/// <https://github.com/gakonst/ethers-rs/blob/ba00f549/ethers-signers/src/wallet/private_key.rs#L197>
#[test]
fn ethereum_end_to_end_example() {
    let signing_key = SigningKey::from_bytes(
        &hex!("4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318").into(),
    )
    .unwrap();

    let msg = hex!(
        "e9808504e3b29200831e848094f0109fc8df283027b6285cc889f5aa624eac1f55843b9aca0080018080"
    );
    let digest = Keccak256::new_with_prefix(msg);

    let (sig, recid) = signing_key.sign_digest_recoverable(digest.clone()).unwrap();
    assert_eq!(
        sig.to_bytes().as_slice(),
        &hex!(
            "c9cf86333bcb065d140032ecaab5d9281bde80f21b9687b3e94161de42d51895727a108a0b8d101465414033c3f705a9c7b826e596766046ee1183dbc8aeaa68"
        )
    );
    assert_eq!(recid, RecoveryId::from_byte(0).unwrap());

    let verifying_key = VerifyingKey::recover_from_digest(digest.clone(), &sig, recid).unwrap();

    assert_eq!(signing_key.verifying_key(), &verifying_key);
    assert!(verifying_key.verify_digest(digest, &sig).is_ok());
}
