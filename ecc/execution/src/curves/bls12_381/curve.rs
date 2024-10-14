use halo2curves_axiom::bls12_381::{Fq, Fq2, G1Affine, G2Affine};
use lazy_static::lazy_static;
use num::{BigInt, Num};
use rand::Rng;

use crate::common::{AffineCoords, FieldExtension};

lazy_static! {
    // polyFactor = (1-x)/3
    pub static ref POLY_FACTOR: BigInt = BigInt::from_str_radix("5044125407647214251", 10).unwrap();

    // finalExpFactor = ((q^12 - 1) / r) / (27 * polyFactor)
    pub static ref FINAL_EXP_FACTOR: BigInt = BigInt::from_str_radix("2366356426548243601069753987687709088104621721678962410379583120840019275952471579477684846670499039076873213559162845121989217658133790336552276567078487633052653005423051750848782286407340332979263075575489766963251914185767058009683318020965829271737924625612375201545022326908440428522712877494557944965298566001441468676802477524234094954960009227631543471415676620753242466901942121887152806837594306028649150255258504417829961387165043999299071444887652375514277477719817175923289019181393803729926249507024121957184340179467502106891835144220611408665090353102353194448552304429530104218473070114105759487413726485729058069746063140422361472585604626055492939586602274983146215294625774144156395553405525711143696689756441298365274341189385646499074862712688473936093315628166094221735056483459332831845007196600723053356837526749543765815988577005929923802636375670820616189737737304893769679803809426304143627363860243558537831172903494450556755190448279875942974830469855835666815454271389438587399739607656399812689280234103023464545891697941661992848552456326290792224091557256350095392859243101357349751064730561345062266850238821755009430903520645523345000326783803935359711318798844368754833295302563158150573540616830138810935344206231367357992991289265295323280", 10).unwrap();

    // lambda = q - u, the optimal exponent
    pub static ref LAMBDA: BigInt = BigInt::from_str_radix("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129030796414117214202539", 10).unwrap();

    // t = -u = 15132376222941642752
    pub static ref T: BigInt = BigInt::from_str_radix("15132376222941642752", 10).unwrap();
}

// BLS12-381 pseudo-binary encoding
// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bls12381/pairing.go#L322
pub const BLS12_381_SEED: u64 = 0xd201000000010000;
pub const BLS12_381_SEED_NEG: bool = true;
pub const BLS12_381_PBE_BITS: usize = 64;
pub const BLS12_381_PBE: [i8; BLS12_381_PBE_BITS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
];

pub struct Bls12_381;

impl Bls12_381 {
    pub fn xi() -> Fq2 {
        Fq2::from_coeffs(&[Fq::one(), Fq::one()])
    }

    pub fn seed() -> u64 {
        BLS12_381_SEED
    }

    pub fn pseudo_binary_encoding() -> [i8; BLS12_381_PBE_BITS] {
        BLS12_381_PBE
    }
}

impl AffineCoords<Fq> for G1Affine {
    fn x(&self) -> Fq {
        self.x
    }

    fn y(&self) -> Fq {
        self.y
    }

    fn neg(&self) -> Self {
        let mut pt = *self;
        pt.y = -pt.y;
        pt
    }

    fn random(rng: &mut impl Rng) -> Self {
        G1Affine::random(rng)
    }

    fn generator() -> Self {
        G1Affine::generator()
    }
}

impl AffineCoords<Fq2> for G2Affine {
    fn x(&self) -> Fq2 {
        self.x
    }

    fn y(&self) -> Fq2 {
        self.y
    }

    fn neg(&self) -> Self {
        let mut pt = *self;
        pt.y = -pt.y;
        pt
    }

    fn random(rng: &mut impl Rng) -> Self {
        G2Affine::random(rng)
    }

    fn generator() -> Self {
        G2Affine::generator()
    }
}
