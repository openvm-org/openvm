use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::BaseAluImmOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

/// Core columns for bitwise operations with a signed 12-bit immediate.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitwiseLogicImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    /// The low byte and bits `[10:8]` of the signed 12-bit immediate.
    pub c_low: [T; 2],
    /// Sign bit of the immediate.
    pub imm_sign: T,

    pub opcode_xor_flag: T,
    pub opcode_or_flag: T,
    pub opcode_and_flag: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitwiseLogicImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct BitwiseLogicImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BitwiseLogicImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BitwiseLogicImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BitwiseLogicImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BitwiseLogicImmCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitwiseLogicImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_xor_flag,
            cols.opcode_or_flag,
            cols.opcode_and_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.imm_sign);

        // c_low[1] + 0xf8 is a byte iff c_low[1] fits in 3 bits. c_low[0] is
        // range-checked directly.
        self.bus
            .send_range(cols.c_low[0], cols.c_low[1] + AB::Expr::from_u32(0xf8))
            .eval(builder, is_valid.clone());

        // Sign-extended byte limbs of the immediate, as expressions.
        let sign_byte = cols.imm_sign * AB::Expr::from_u32((1 << LIMB_BITS) - 1);
        let c: [AB::Expr; NUM_LIMBS] = array::from_fn(|i| match i {
            0 => cols.c_low[0].into(),
            1 => cols.c_low[1] + cols.imm_sign * AB::Expr::from_u32(0xf8),
            _ => sign_byte.clone(),
        });

        let a = &cols.a;
        let b = &cols.b;

        for i in 0..NUM_LIMBS {
            let x_xor_y = cols.opcode_xor_flag * a[i]
                + cols.opcode_or_flag * ((AB::Expr::from_u32(2) * a[i]) - b[i] - c[i].clone())
                + cols.opcode_and_flag * (b[i] + c[i].clone() - (AB::Expr::from_u32(2) * a[i]));
            self.bus
                .send_xor(b[i], c[i].clone(), x_xor_y)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            cols.opcode_xor_flag * AB::Expr::from_u8(BaseAluImmOpcode::XORI as u8)
                + cols.opcode_or_flag * AB::Expr::from_u8(BaseAluImmOpcode::ORI as u8)
                + cols.opcode_and_flag * AB::Expr::from_u8(BaseAluImmOpcode::ANDI as u8),
        );

        // Canonical 24-bit sign extension of the signed 12-bit immediate.
        let imm = cols.c_low[0]
            + cols.c_low[1] * AB::Expr::from_u32(1 << LIMB_BITS)
            + cols.imm_sign * AB::Expr::from_u32(0xff_f800);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitwiseLogicImmCoreExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct BitwiseLogicImmFiller<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}
