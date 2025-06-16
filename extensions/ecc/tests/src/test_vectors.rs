use hex_literal::hex;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, Bytes};

#[repr(C)]
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryTestVector {
    #[serde_as(as = "Bytes")]
    pub pk: [u8; 33],
    #[serde_as(as = "Bytes")]
    pub msg: [u8; 32],
    #[serde_as(as = "Bytes")]
    pub sig: [u8; 64],
    pub recid: u8,
    pub ok: bool,
}

pub const P256_RECOVERY_TEST_VECTORS: &[RecoveryTestVector] = &[RecoveryTestVector {pk:hex!("020000000000000000000000000000000000000000000000000000000000000000"),msg:hex!("00000000000000000000FFFFFFFF03030BFFFFFFFFFF030BFFFFFFFFFFFFF8FC"),sig:hex!("00000000ffffffff00000000000000004319055258e8617b0c46353d039cdaaf0000000000000000000000000000000000000000000000000000000000000001"),recid:2,ok:false,
},
RecoveryTestVector{pk:hex!("020000000000000000000000000000000000000000000000000000000000000000"),msg:hex!("000000000000000000000000000000000000000000000000000CFD5E267CBB5E"),sig:hex!("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296000000000000000000000000000000000000000000000000000cfd5e267cbb5e"),recid:1,ok:false
}];

pub const K256_RECOVERY_TEST_VECTORS: &[RecoveryTestVector] = &[RecoveryTestVector{pk:hex!("020000000000000000000000000000000000000000000000000000000000000000"),msg:hex!("0000000000000000000000000000000000000000000000000000000000000000"),sig:hex!("0000000000000000000000000000000000000000000000000000000000000001ffffffffbffffffffffffffffeffbaffaeff6f7000000100000000dbd0364140"),recid:1,ok:false}
];

#[repr(C)]
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sec1DecodingTestVector {
    #[serde_as(as = "Bytes")]
    pub bytes: Vec<u8>,
    pub ok: bool,
}

pub fn k256_sec1_decoding_test_vectors() -> Vec<Sec1DecodingTestVector> {
    vec![
    Sec1DecodingTestVector {
        bytes: hex!("04" "0000000000000000000000000000000000000000000000000000000000000000" "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f").to_vec(),
        ok: false,
    },
    Sec1DecodingTestVector {
        bytes: hex!("04" "0000000000000000000000000000000000000000000000000000000000000001" "4218f20ae6c646b363db68605822fb14264ca8d2587fdd6fbc750d587e76a7ee").to_vec(),
        ok: true,
    },
    Sec1DecodingTestVector {
        bytes: hex!("04" "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc30" "4218f20ae6c646b363db68605822fb14264ca8d2587fdd6fbc750d587e76a7ee").to_vec(),
        ok: false,
    },
    Sec1DecodingTestVector {
        bytes: hex!("04" "0000000000000000000000000000000000000000000000000000000000000000" "0000000000000000000000000000000000000000000000000000000000000000").to_vec(),
        ok: false,
    }
]
}
