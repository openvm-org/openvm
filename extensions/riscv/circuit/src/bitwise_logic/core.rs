use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitwiseLogicCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_xor_flag: T,
    pub opcode_or_flag: T,
    pub opcode_and_flag: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitwiseLogicCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct BitwiseLogicCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BitwiseLogicCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BitwiseLogicCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BitwiseLogicCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BitwiseLogicCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc_idx: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitwiseLogicCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
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

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // Interaction with BitwiseOperationLookup to constrain a's correctness for XOR, OR,
        // and AND. The expected b^c is encoded algebraically per opcode:
        //   XOR: b^c = a
        //   OR:  b^c = 2a - b - c
        //   AND: b^c = b + c - 2a
        // This also range checks b and c; a is then uniquely determined within byte range.
        for i in 0..NUM_LIMBS {
            let x_xor_y = cols.opcode_xor_flag * a[i]
                + cols.opcode_or_flag * ((AB::Expr::from_u32(2) * a[i]) - b[i] - c[i])
                + cols.opcode_and_flag * (b[i] + c[i] - (AB::Expr::from_u32(2) * a[i]));
            self.bus
                .send_xor(b[i], c[i], x_xor_y)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            cols.opcode_xor_flag * AB::Expr::from_u8(BaseAluOpcode::XOR as u8)
                + cols.opcode_or_flag * AB::Expr::from_u8(BaseAluOpcode::OR as u8)
                + cols.opcode_and_flag * AB::Expr::from_u8(BaseAluOpcode::AND as u8),
        );

        AdapterAirContext {
            to_pc_idx: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitwiseLogicCoreExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct BitwiseLogicFiller<const LIMB_BITS: usize> {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

#[inline(always)]
pub(crate) fn run_bitwise_logic<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    debug_assert!(LIMB_BITS <= 8, "specialize for bytes");
    match opcode {
        BaseAluOpcode::XOR => run_xor::<NUM_LIMBS>(x, y),
        BaseAluOpcode::OR => run_or::<NUM_LIMBS>(x, y),
        BaseAluOpcode::AND => run_and::<NUM_LIMBS>(x, y),
        _ => unreachable!("BitwiseLogicCoreExecutor received non-XOR/OR/AND opcode"),
    }
}

#[inline(always)]
fn run_xor<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] ^ y[i])
}

#[inline(always)]
fn run_or<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] | y[i])
}

#[inline(always)]
fn run_and<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] & y[i])
}
