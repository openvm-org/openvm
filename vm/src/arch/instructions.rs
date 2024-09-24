use std::{collections::BTreeMap, fmt, iter};

use afs_stark_backend::interaction::InteractionBuilder;
use enum_utils::FromStr;
use p3_air::FilteredAirBuilder;
use p3_field::{AbstractField, Field};
use Opcode::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromStr, PartialOrd, Ord)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum Opcode {
    NOP = 0,

    LOADW = 1,
    STOREW = 2,
    LOADW2 = 3,
    STOREW2 = 4,
    JAL = 5,
    BEQ = 6,
    BNE = 7,
    TERMINATE = 8,
    PUBLISH = 9,

    FADD = 10,
    FSUB = 11,
    FMUL = 12,
    FDIV = 13,

    CASTF = 14,

    FAIL = 20,
    PRINTF = 21,

    FE4ADD = 30,
    FE4SUB = 31,
    BBE4MUL = 32,
    BBE4DIV = 33,

    PERM_POS2 = 40,
    COMP_POS2 = 41,
    KECCAK256 = 42,

    /// Instruction to write the next hint word into memory.
    SHINTW = 50,

    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT = 51,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS = 52,
    /// Phantom instruction to prepare the little-endian byte decomposition of a variable for hinting.
    HINT_BYTES = 53,

    /// Phantom instruction to start tracing
    CT_START = 60,
    /// Phantom instruction to end tracing
    CT_END = 61,

    SECP256K1_COORD_ADD = 70,
    SECP256K1_COORD_SUB = 71,
    SECP256K1_COORD_MUL = 72,
    SECP256K1_COORD_DIV = 73,

    SECP256K1_SCALAR_ADD = 74,
    SECP256K1_SCALAR_SUB = 75,
    SECP256K1_SCALAR_MUL = 76,
    SECP256K1_SCALAR_DIV = 77,

    ADD256 = 80,
    SUB256 = 81,
    MUL256 = 82,
    LT256 = 83,
    EQ256 = 84,
    XOR256 = 85,
    AND256 = 86,
    OR256 = 87,
    SLT256 = 88,
    SLL256 = 89,
    SRL256 = 90,
    SRA256 = 91,

    LUI = 98,
    AUIPC = 99,

    SECP256K1_EC_ADD_NE = 101,
    SECP256K1_EC_DOUBLE = 102,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub const CORE_INSTRUCTIONS: [Opcode; 17] = [
    LOADW, STOREW, JAL, BEQ, BNE, TERMINATE, PRINTF, SHINTW, HINT_INPUT, HINT_BITS, HINT_BYTES,
    PUBLISH, CT_START, CT_END, NOP, LOADW2, STOREW2,
];
pub const FIELD_ARITHMETIC_INSTRUCTIONS: [Opcode; 4] = [FADD, FSUB, FMUL, FDIV];
pub const FIELD_EXTENSION_INSTRUCTIONS: [Opcode; 4] = [FE4ADD, FE4SUB, BBE4MUL, BBE4DIV];
pub const ALU_256_INSTRUCTIONS: [Opcode; 8] =
    [ADD256, SUB256, LT256, EQ256, XOR256, AND256, OR256, SLT256];
pub const SHIFT_256_INSTRUCTIONS: [Opcode; 3] = [SLL256, SRL256, SRA256];
pub const UI_32_INSTRUCTIONS: [Opcode; 2] = [LUI, AUIPC];

pub const MODULAR_ADDSUB_INSTRUCTIONS: [Opcode; 4] = [
    SECP256K1_COORD_ADD,
    SECP256K1_COORD_SUB,
    SECP256K1_SCALAR_ADD,
    SECP256K1_SCALAR_SUB,
];

pub const MODULAR_MULTDIV_INSTRUCTIONS: [Opcode; 4] = [
    SECP256K1_COORD_MUL,
    SECP256K1_COORD_DIV,
    SECP256K1_SCALAR_MUL,
    SECP256K1_SCALAR_DIV,
];

impl Opcode {
    pub fn all_opcodes() -> Vec<Opcode> {
        let mut all_opcodes = vec![];
        all_opcodes.extend(CORE_INSTRUCTIONS);
        all_opcodes.extend(FIELD_ARITHMETIC_INSTRUCTIONS);
        all_opcodes.extend(FIELD_EXTENSION_INSTRUCTIONS);
        all_opcodes.extend([FAIL]);
        all_opcodes.extend([PERM_POS2, COMP_POS2]);
        all_opcodes
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        Self::all_opcodes()
            .into_iter()
            .find(|&opcode| value == opcode as u8)
    }
}

#[derive(Clone, Debug)]
pub struct OpcodeEncoder<const N: usize> {
    coords_map: BTreeMap<Opcode, [usize; N]>,
}

impl<const N: usize> OpcodeEncoder<N> {
    pub fn new(opcodes: impl IntoIterator<Item = Opcode>) -> Self {
        let opcodes_with_nop = iter::once(NOP).chain(opcodes);
        let mut coords = [0; N];
        let mut coords_map = BTreeMap::new();
        for opcode in opcodes_with_nop {
            assert!(coords[0] <= 2, "too many opcodes");
            coords_map.insert(opcode, coords);

            let mut i = N - 1;
            while i > 0 && coords[i] == 2 {
                i -= 1;
            }
            coords[i] += 1;
            for x in coords.iter_mut().skip(i + 1) {
                *x = 0;
            }
        }

        Self { coords_map }
    }

    pub fn initialize<AB: InteractionBuilder>(&self, builder: &mut AB, variables: [AB::Var; N]) {
        for &v in variables.iter() {
            builder.assert_zero(v * (v - AB::Expr::one()) * (v - AB::Expr::two()));
        }
        let sum = variables.iter().fold(AB::Expr::zero(), |acc, x| acc + (*x));
        builder.assert_zero(
            sum.clone() * (sum.clone() - AB::Expr::one()) * (sum.clone() - AB::Expr::two()),
        );
    }

    pub fn encode(&self, opcode: Opcode) -> [usize; N] {
        *self.coords_map.get(&opcode).unwrap()
    }

    pub fn expression_for<AB: InteractionBuilder>(
        &self,
        opcode: Opcode,
        variables: [AB::Var; N],
    ) -> AB::Expr {
        let coords = self.coords_map.get(&opcode).unwrap();
        let mut expr = AB::Expr::one();
        // We need to "normalize" the expression so that the value at this point is 1.
        // We don't need it for the "when" condition, but we may want to calculate
        // the opcode as sum(flag * opcode).
        let mut denom = AB::F::one();
        for i in 0..N {
            for j in 0..coords[i] {
                expr *= variables[i] - AB::Expr::from_canonical_usize(j);
                denom *= AB::F::from_canonical_usize(coords[i] - j);
            }
        }
        let sum = variables.iter().fold(AB::Expr::zero(), |acc, x| acc + (*x));
        let sum_coords = coords.iter().sum::<usize>();
        for j in sum_coords + 1..=2 {
            expr *= AB::Expr::from_canonical_usize(j) - sum.clone();
            denom *= AB::F::from_canonical_usize(j - sum_coords);
        }
        expr * denom.inverse()
    }

    pub fn when<'a, AB: InteractionBuilder>(
        &self,
        builder: &'a mut AB,
        variables: [AB::Var; N],
        opcode: Opcode,
    ) -> FilteredAirBuilder<'a, AB> {
        builder.when(self.expression_for::<AB>(opcode, variables))
    }
}
