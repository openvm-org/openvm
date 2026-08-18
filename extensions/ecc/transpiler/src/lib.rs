use openvm_decoder::instruction_formats::RType;
use openvm_ecc_guest::{SwBaseFunct7, OPCODE, SW_FUNCT3};
use openvm_instructions::{
    instruction::Instruction, riscv::REGISTER_NUM_LIMBS, LocalOpcode, VmOpcode,
};
use openvm_instructions_derive::LocalOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{util::from_r_type, TranspilerExtension, TranspilerOutput};
use strum::{EnumCount, EnumIter, FromRepr};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x600]
#[allow(non_camel_case_types)]
#[repr(usize)]
/// Short Weierstrass curve operations.
///
/// These operations use partial affine formulas. The caller must meet each requirement below.
pub enum WeierstrassOpcode {
    /// Adds two affine points.
    ///
    /// Requirements:
    /// - Each point must be on the curve.
    ///   - The operation uses the curve addition formula.
    /// - Each point must not be the identity.
    ///   - The identity has no affine coordinates.
    /// - The x-coordinates must be different.
    ///   - The formula divides by `x2 - x1`.
    EC_ADD_NE,
    SETUP_EC_ADD_NE,
    /// Doubles an affine point.
    ///
    /// Requirements:
    /// - The input point must be on the curve.
    ///   - The operation uses the curve doubling formula.
    /// - The input point must not be the identity.
    ///   - The identity has no affine coordinates.
    /// - The result must not be the identity.
    ///   - The formula divides by `2 * y`, which is zero when the result is the identity.
    EC_DOUBLE,
    SETUP_EC_DOUBLE,
    /// Multiplies an affine point by a scalar.
    ///
    /// Requirements:
    /// - The point must be on the curve.
    ///   - The ladder uses the curve addition and doubling formulas.
    /// - The point must not be the identity.
    ///   - The ladder starts with the point and uses affine coordinates.
    /// - The point must be in a subgroup of prime order `n`.
    ///   - A nonzero multiple less than `n` is not the identity.
    /// - The scalar `k` must be odd and less than `n`.
    ///   - The ladder writes `k` as `2 * B + 1`. Each step adds `P` or `-P`.
    /// - The subgroup order `n` must equal 1 modulo 4.
    ///   - This prevents an addition of points with equal x-coordinates during the ladder.
    EC_MUL,
    SETUP_EC_MUL,
}

const _: () =
    assert!(WeierstrassOpcode::COUNT <= SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize);

#[derive(Default)]
pub struct EccTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for EccTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream.is_empty() {
            return None;
        }
        let instruction_u32 = instruction_stream[0];
        let opcode = (instruction_u32 & 0x7f) as u8;
        let funct3 = ((instruction_u32 >> 12) & 0b111) as u8;

        if opcode != OPCODE {
            return None;
        }
        if funct3 != SW_FUNCT3 {
            return None;
        }

        let instruction = {
            // short weierstrass ec
            let dec_insn = RType::new(instruction_u32);
            let base_funct7 = (dec_insn.funct7 as u8) % SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS;
            let curve_idx =
                ((dec_insn.funct7 as u8) / SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS) as usize;
            let curve_idx_shift = curve_idx * WeierstrassOpcode::COUNT;
            if base_funct7 == SwBaseFunct7::SwSetup as u8
                || base_funct7 == SwBaseFunct7::SwSetupMul as u8
            {
                let local_opcode = if base_funct7 == SwBaseFunct7::SwSetupMul as u8 {
                    WeierstrassOpcode::SETUP_EC_MUL
                } else {
                    match dec_insn.rs2 {
                        0 => WeierstrassOpcode::SETUP_EC_DOUBLE,
                        _ => WeierstrassOpcode::SETUP_EC_ADD_NE,
                    }
                };
                Some(Instruction::new(
                    VmOpcode::from_usize(local_opcode.global_opcode().as_usize() + curve_idx_shift),
                    F::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
                    F::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
                    F::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs2),
                    F::ONE, // d_as = 1
                    F::TWO, // e_as = 2
                    F::ZERO,
                    F::ZERO,
                ))
            } else {
                let global_opcode = match SwBaseFunct7::from_repr(base_funct7) {
                    Some(SwBaseFunct7::SwAddNe) => {
                        WeierstrassOpcode::EC_ADD_NE as usize + WeierstrassOpcode::CLASS_OFFSET
                    }
                    Some(SwBaseFunct7::SwDouble) => {
                        assert!(dec_insn.rs2 == 0);
                        WeierstrassOpcode::EC_DOUBLE as usize + WeierstrassOpcode::CLASS_OFFSET
                    }
                    Some(SwBaseFunct7::SwMul) => {
                        WeierstrassOpcode::EC_MUL as usize + WeierstrassOpcode::CLASS_OFFSET
                    }
                    _ => unimplemented!(),
                };
                let global_opcode = global_opcode + curve_idx_shift;
                Some(from_r_type(global_opcode, 2, &dec_insn, true))
            }
        };
        instruction.map(TranspilerOutput::one_to_one)
    }
}
