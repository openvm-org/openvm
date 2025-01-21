use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, ImmInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::NativeJalOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JalCoreCols<T> {
    pub imm: T,
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug)]
pub struct JalCoreAir {}

impl<F: Field> BaseAir<F> for JalCoreAir {
    fn width(&self) -> usize {
        JalCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JalCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for JalCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 0]>,
    I::Writes: From<[[AB::Expr; 1]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JalCoreCols<_> = local_core.borrow();

        AdapterAirContext {
            to_pc: Some(from_pc.into() + cols.imm.into()),
            reads: [].into(),
            writes: [[from_pc.into() + AB::Expr::from_canonical_u32(DEFAULT_PC_STEP)]].into(),
            instruction: ImmInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_canonical_usize(
                    NativeJalOpcode::JAL.global_opcode().as_usize(),
                ),
                immediate: cols.imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        NativeJalOpcode::CLASS_OFFSET
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JalRecord<F> {
    pub imm: F,
}

pub struct JalCoreChip {
    pub air: JalCoreAir,
}

impl JalCoreChip {
    pub fn new() -> Self {
        Self { air: JalCoreAir {} }
    }
}

impl Default for JalCoreChip {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for JalCoreChip
where
    I::Reads: From<[[F; 1]; 0]>,
    I::Writes: From<[[F; 1]; 1]>,
{
    type Record = JalRecord<F>;
    type Air = JalCoreAir;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, b, .. } = instruction;
        assert_eq!(
            NativeJalOpcode::from_usize(opcode.local_opcode_idx(NativeJalOpcode::CLASS_OFFSET)),
            NativeJalOpcode::JAL
        );

        let output = AdapterRuntimeContext {
            to_pc: Some((F::from_canonical_u32(from_pc) + *b).as_canonical_u32()),
            writes: [[F::from_canonical_u32(from_pc + DEFAULT_PC_STEP)]].into(),
        };

        Ok((output, JalRecord { imm: *b }))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeJalOpcode::from_usize(opcode - NativeJalOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let JalRecord { imm } = record;
        let row_slice: &mut JalCoreCols<_> = row_slice.borrow_mut();
        row_slice.imm = imm;
        row_slice.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
