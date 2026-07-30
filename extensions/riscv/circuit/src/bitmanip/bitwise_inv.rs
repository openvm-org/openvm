use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use super::core::{is_bitwise_inv_opcode, run_bitmanip_reg, ANDN, BITMANIP_OFFSET, ORN, XNOR};
use crate::adapters::RV64_BYTE_BITS;

/// ANDN/ORN/XNOR over byte limbs via the bitwise XOR lookup, with the
/// complement of `c` folded linearly into the lookup relation (`~c = 255 - c`
/// per byte), mirroring `BitwiseLogicCoreAir`:
///   ANDN: b ^ ~c = b + ~c - 2a
///   ORN:  b ^ ~c = 2a - b - ~c
///   XNOR: b ^ c  = 255 - a
/// The lookup range checks its inputs, which bounds `c` (via `~c`) and `a`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipBitwiseInvCoreCols<T> {
    pub a: [T; RV64_REGISTER_NUM_LIMBS],
    pub b: [T; RV64_REGISTER_NUM_LIMBS],
    pub c: [T; RV64_REGISTER_NUM_LIMBS],

    pub opcode_andn_flag: T,
    pub opcode_orn_flag: T,
    pub opcode_xnor_flag: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipBitwiseInvCoreCols<u8>)]
pub struct BitManipBitwiseInvCoreAir {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for BitManipBitwiseInvCoreAir {
    fn width(&self) -> usize {
        BitManipBitwiseInvCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipBitwiseInvCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipBitwiseInvCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipBitwiseInvCoreCols<_> = local_core.borrow();
        let flags = [
            (cols.opcode_andn_flag, ANDN),
            (cols.opcode_orn_flag, ORN),
            (cols.opcode_xnor_flag, XNOR),
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &(flag, _)| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let max_byte = AB::Expr::from_u8(u8::MAX);
        for i in 0..RV64_REGISTER_NUM_LIMBS {
            let a = cols.a[i];
            let b = cols.b[i];
            let c = cols.c[i];
            let c_inv = max_byte.clone() - c;

            // The second lookup input is ~c for ANDN/ORN and c for XNOR.
            let y = (cols.opcode_andn_flag + cols.opcode_orn_flag) * c_inv.clone()
                + cols.opcode_xnor_flag * c;
            // The expected XOR of the two lookup inputs, encoded per opcode.
            let x_xor_y = cols.opcode_andn_flag * (b + c_inv.clone() - AB::Expr::from_u8(2) * a)
                + cols.opcode_orn_flag * (AB::Expr::from_u8(2) * a - b - c_inv)
                + cols.opcode_xnor_flag * (max_byte.clone() - a);
            self.bus
                .send_xor(b.into(), y, x_xor_y)
                .eval(builder, is_valid.clone());
        }

        let expected_local_opcode = flags
            .iter()
            .fold(AB::Expr::ZERO, |acc, &(flag, local_opcode)| {
                acc + flag * AB::Expr::from_usize(local_opcode)
            });
        let expected_opcode = expected_local_opcode + AB::Expr::from_usize(BITMANIP_OFFSET);

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
        BITMANIP_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipBitwiseInvCoreRecord {
    pub b: [u8; RV64_REGISTER_NUM_LIMBS],
    pub c: [u8; RV64_REGISTER_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipBitwiseInvExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipBitwiseInvFiller<A> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<{ RV64_BYTE_BITS }>,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipBitwiseInvExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; RV64_REGISTER_NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; RV64_REGISTER_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BitManipBitwiseInvCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BBitwiseInv({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_bitwise_inv_opcode(local_opcode));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.local_opcode = local_opcode as u8;

        let output = run_bitmanip_reg(
            local_opcode,
            u64::from_le_bytes(core_record.b),
            u64::from_le_bytes(core_record.c),
        );
        self.adapter.write(
            state.memory,
            instruction,
            [output.to_le_bytes()].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for BitManipBitwiseInvFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipBitwiseInvCoreRecord =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let c = record.c;
        let local_opcode = record.local_opcode as usize;
        let a = run_bitmanip_reg(local_opcode, u64::from_le_bytes(b), u64::from_le_bytes(c))
            .to_le_bytes();

        for i in 0..RV64_REGISTER_NUM_LIMBS {
            if local_opcode == XNOR {
                self.bitwise_lookup_chip
                    .request_xor(b[i] as u32, c[i] as u32);
            } else {
                self.bitwise_lookup_chip
                    .request_xor(b[i] as u32, (u8::MAX - c[i]) as u32);
            }
        }

        let core_row: &mut BitManipBitwiseInvCoreCols<F> = core_row.borrow_mut();
        core_row.opcode_xnor_flag = F::from_bool(local_opcode == XNOR);
        core_row.opcode_orn_flag = F::from_bool(local_opcode == ORN);
        core_row.opcode_andn_flag = F::from_bool(local_opcode == ANDN);
        core_row.c = c.map(F::from_u8);
        core_row.b = b.map(F::from_u8);
        core_row.a = a.map(F::from_u8);
    }
}
