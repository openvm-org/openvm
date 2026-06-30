use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_riscv_transpiler::AddIOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct AddICoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(AddICoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct AddICoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for AddICoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        AddICoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for AddICoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for AddICoreAir<NUM_LIMBS, LIMB_BITS>
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
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &AddICoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();

        let is_valid: AB::Expr = cols.is_valid.into();
        builder.assert_bool(cols.is_valid);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (b[i] + c[i] - a[i]
                    + if i > 0 {
                        carry[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder.when(cols.is_valid).assert_bool(carry[i].clone());
        }

        for &a_limb in a.iter() {
            self.range_bus
                .range_check(a_limb, LIMB_BITS)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_u8(AddIOpcode::ADDI as u8),
        );

        AdapterAirContext {
            to_pc: None,
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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct AddICoreRecord<const NUM_LIMBS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub c: [u16; NUM_LIMBS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct AddIExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct AddIFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for AddIExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; NUM_LIMBS]; 2]>,
            WriteData: From<[[u16; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut AddICoreRecord<NUM_LIMBS>),
    >,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", AddIOpcode::ADDI)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = run_add::<NUM_LIMBS, LIMB_BITS>(&core_record.b, &core_record.c);

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for AddIFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &AddICoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut AddICoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let a = run_add::<NUM_LIMBS, LIMB_BITS>(&record.b, &record.c);
        core_row.is_valid = F::ONE;
        for a_val in a {
            self.range_checker_chip.add_count(a_val as u32, LIMB_BITS);
        }
        core_row.c = record.c.map(F::from_u16);
        core_row.b = record.b.map(F::from_u16);
        core_row.a = a.map(F::from_u16);
    }
}

#[inline(always)]
fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> [u16; NUM_LIMBS] {
    let mut z = [0u16; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut overflow = (x[i] as u32) + (y[i] as u32) + if i > 0 { carry[i - 1] } else { 0 };
        carry[i] = overflow >> LIMB_BITS;
        overflow &= (1u32 << LIMB_BITS) - 1;
        z[i] = overflow as u16;
    }
    z
}
