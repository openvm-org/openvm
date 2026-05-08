//! Algebra extension for rvr-openvm.
//!
//! Provides IR nodes for modular arithmetic (ADD, SUB, MUL, DIV, IS_EQ, SETUP),
//! Fp2 (complex extension field) operations, and phantom instructions
//! (HintNonQr, HintSqrt). Lifting/codegen splits across [`ModularRvrExtension`]
//! (modular + phantoms; ships the lift-time C and libsecp256k1 inputs for
//! k256) and [`Fp2RvrExtension`] (fp2 ops only; Rust-only).

mod fp2;
mod modular;

pub use fp2::{Fp2ArithInstr, Fp2RvrExtension, Fp2SetupInstr};
pub use modular::{
    HintNonQrInstr, HintSqrtInstr, ModArithInstr, ModIsEqInstr, ModSetupInstr, ModularRvrExtension,
};
use num_bigint::BigUint;
use openvm_algebra_utils::find_non_qr;
use rand::{rngs::StdRng, SeedableRng};

// ── Modular arithmetic operations ────────────────────────────────────────────

/// Operation type for modular arithmetic.
#[derive(Debug, Clone, Copy)]
pub enum ModOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ModOp {
    /// Lower-case op name used as a suffix in the generated C function name
    /// (e.g. `rvr_ext_mod_add`, `rvr_ext_mod_sub_k256_coord`).
    pub(crate) fn c_name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }
}

// ── Known curve detection for native field arithmetic ─────────────────────────

/// Known field types that have optimized native FFI implementations.
#[derive(Debug, Clone, Copy)]
pub(crate) enum KnownField {
    K256Coord,
    K256Scalar,
    P256Coord,
    P256Scalar,
    Bn254Fq,
    Bn254Fr,
    Bls12381Fq,
    Bls12381Fr,
}

impl KnownField {
    /// C function name suffix for this field.
    pub(crate) fn c_suffix(self) -> &'static str {
        match self {
            Self::K256Coord => "k256_coord",
            Self::K256Scalar => "k256_scalar",
            Self::P256Coord => "p256_coord",
            Self::P256Scalar => "p256_scalar",
            Self::Bn254Fq => "bn254_fq",
            Self::Bn254Fr => "bn254_fr",
            Self::Bls12381Fq => "bls12_381_fq",
            Self::Bls12381Fr => "bls12_381_fr",
        }
    }

    /// Fp2 C function name suffix (only valid for base fields of Fp2-capable curves).
    pub(crate) fn fp2_c_suffix(self) -> Option<&'static str> {
        match self {
            Self::Bn254Fq => Some("bn254"),
            Self::Bls12381Fq => Some("bls12_381"),
            _ => None,
        }
    }
}

/// Known moduli (LE, padded) mapped to their field types.
static KNOWN_FIELDS: &[(&[u8], KnownField)] = &[
    // secp256k1 coordinate field
    (
        &[
            0x2f, 0xfc, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::K256Coord,
    ),
    // secp256k1 scalar field
    (
        &[
            0x41, 0x41, 0x36, 0xd0, 0x8c, 0x5e, 0xd2, 0xbf, 0x3b, 0xa0, 0x48, 0xaf, 0xe6, 0xdc,
            0xae, 0xba, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::K256Scalar,
    ),
    // secp256r1 coordinate field
    (
        &[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::P256Coord,
    ),
    // secp256r1 scalar field
    (
        &[
            0x51, 0x25, 0x63, 0xfc, 0xc2, 0xca, 0xb9, 0xf3, 0x84, 0x9e, 0x17, 0xa7, 0xad, 0xfa,
            0xe6, 0xbc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::P256Scalar,
    ),
    // BN254 base field
    (
        &[
            0x47, 0xfd, 0x7c, 0xd8, 0x16, 0x8c, 0x20, 0x3c, 0x8d, 0xca, 0x71, 0x68, 0x91, 0x6a,
            0x81, 0x97, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1,
            0x72, 0x4e, 0x64, 0x30,
        ],
        KnownField::Bn254Fq,
    ),
    // BN254 scalar field
    (
        &[
            0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9, 0x79, 0x48, 0xe8,
            0x33, 0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1,
            0x72, 0x4e, 0x64, 0x30,
        ],
        KnownField::Bn254Fr,
    ),
    // BLS12-381 base field (48 bytes)
    (
        &[
            0xab, 0xaa, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xb9, 0xff, 0xff, 0x53, 0xb1, 0xfe, 0xff,
            0xab, 0x1e, 0x24, 0xf6, 0xb0, 0xf6, 0xa0, 0xd2, 0x30, 0x67, 0xbf, 0x12, 0x85, 0xf3,
            0x84, 0x4b, 0x77, 0x64, 0xd7, 0xac, 0x4b, 0x43, 0xb6, 0xa7, 0x1b, 0x4b, 0x9a, 0xe6,
            0x7f, 0x39, 0xea, 0x11, 0x01, 0x1a,
        ],
        KnownField::Bls12381Fq,
    ),
    // BLS12-381 scalar field
    (
        &[
            0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x5b, 0xfe, 0xff, 0x02, 0xa4,
            0xbd, 0x53, 0x05, 0xd8, 0xa1, 0x09, 0x08, 0xd8, 0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29,
            0x53, 0xa7, 0xed, 0x73,
        ],
        KnownField::Bls12381Fr,
    ),
];

/// Detect a known field from its modulus bytes (LE, padded).
pub(crate) fn detect_known_field(modulus_bytes: &[u8]) -> Option<KnownField> {
    KNOWN_FIELDS
        .iter()
        .find(|(bytes, _)| *bytes == modulus_bytes)
        .map(|(_, f)| *f)
}

// ── Shared infrastructure ────────────────────────────────────────────────────

/// Per-modulus info shared by the algebra extensions.
pub(crate) struct ModulusInfo {
    pub(crate) modulus_bytes: Vec<u8>,
    pub(crate) non_qr_bytes: Vec<u8>,
    pub(crate) num_limbs: u32,
}

pub(crate) fn make_moduli(moduli: Vec<BigUint>) -> Vec<ModulusInfo> {
    // Use the same deterministic seed as OpenVM for non-QR computation.
    // For ModularRvrExtension this matches the circuit-side
    // `NonQrHintSubEx::new` (also `StdRng::from_seed([0u8; 32])`, single rng
    // across the full modulus list), so rvr-emitted NQRs match what the
    // circuit would compute.
    //
    // TODO: Fp2RvrExtension also calls this, but `try_lift_fp2` never reads
    // `info.non_qr_bytes` — Fp2 NQR computation is dead work today. It also
    // has a latent non-determinism: the same modulus appearing in both
    // `moduli` and `fp2_moduli` is processed by two independent rngs (each
    // freshly seeded here), so for primes that fall through to
    // rejection-sampling in `find_non_qr` (anything not `p ≡ 3 (mod 4)` or
    // `p ≡ 5 (mod 8)`), the NQR for the same prime can differ between lists.
    // If a future Fp2 phantom ever consumes `info.non_qr_bytes`, this will
    // diverge from what the circuit computes. Either drop NQR computation
    // for fp2 (use a `make_fp2_moduli` that fills num_limbs+modulus_bytes
    // only), or share rng state with the modular list.
    let mut rng = StdRng::from_seed([0u8; 32]);
    moduli
        .into_iter()
        .map(|m| make_modulus_info(&m, &mut rng))
        .collect()
}

fn make_modulus_info(modulus: &BigUint, rng: &mut StdRng) -> ModulusInfo {
    let bytes = modulus.bits().div_ceil(8) as usize;
    assert!(
        bytes <= 48,
        "modulus exceeds maximum supported size of 384 bits"
    );
    let num_limbs = if bytes <= 32 { 32u32 } else { 48u32 };
    let mut modulus_bytes = modulus.to_bytes_le();
    modulus_bytes.resize(num_limbs as usize, 0);
    let non_qr = find_non_qr(modulus, rng);
    let mut non_qr_bytes = non_qr.to_bytes_le();
    non_qr_bytes.resize(num_limbs as usize, 0);
    ModulusInfo {
        modulus_bytes,
        non_qr_bytes,
        num_limbs,
    }
}

/// Format a byte slice as a C array initializer: `{0x2f, 0xfc, ...}`
pub(crate) fn format_c_byte_array(bytes: &[u8]) -> String {
    let inner: Vec<String> = bytes.iter().map(|b| format!("0x{b:02x}")).collect();
    format!("{{{}}}", inner.join(","))
}
