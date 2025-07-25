use std::{
    array,
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory, POINTER_MAX_BITS,
    },
};
use openvm_circuit_primitives::{AlignedBorrow, AlignedBytesBorrow};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, NATIVE_AS,
};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{LoadStoreInstruction, Rv32LoadStoreAdapterFiller};

#[derive(Debug, Clone, Copy)]
enum InstructionOpcode {
    LoadW0,
    LoadHu0,
    LoadHu2,
    LoadBu0,
    LoadBu1,
    LoadBu2,
    LoadBu3,
    StoreW0,
    StoreH0,
    StoreH2,
    StoreB0,
    StoreB1,
    StoreB2,
    StoreB3,
}

use InstructionOpcode::*;

/// LoadStore Core Chip handles byte/halfword into word conversions and unsigned extends
/// This chip uses read_data and prev_data to constrain the write_data
/// It also handles the shifting in case of not 4 byte aligned instructions
/// This chips treats each (opcode, shift) pair as a separate instruction
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub flags: [T; 4],
    /// we need to keep the degree of is_valid and is_load to 1
    pub is_valid: T,
    pub is_load: T,

    pub read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
    /// write_data will be constrained against read_data and prev_data
    /// depending on the opcode and the shift amount
    pub write_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone, derive_new::new)]
pub struct LoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for LoadStoreCoreAir<NUM_CELLS> {
    fn width(&self) -> usize {
        LoadStoreCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize> BaseAirWithPublicValues<F> for LoadStoreCoreAir<NUM_CELLS> {}

impl<AB, I, const NUM_CELLS: usize> VmCoreAir<AB, I> for LoadStoreCoreAir<NUM_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<([AB::Var; NUM_CELLS], [AB::Expr; NUM_CELLS])>,
    I::Writes: From<[[AB::Expr; NUM_CELLS]; 1]>,
    I::ProcessedInstruction: From<LoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadStoreCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadStoreCoreCols::<AB::Var, NUM_CELLS> {
            read_data,
            prev_data,
            write_data,
            flags,
            is_valid,
            is_load,
        } = *cols;

        let get_expr_12 = |x: &AB::Expr| (x.clone() - AB::Expr::ONE) * (x.clone() - AB::Expr::TWO);

        builder.assert_bool(is_valid);
        let sum = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_zero(flag * get_expr_12(&flag.into()));
            acc + flag
        });
        builder.assert_zero(sum.clone() * get_expr_12(&sum));
        // when sum is 0, is_valid must be 0
        builder.when(get_expr_12(&sum)).assert_zero(is_valid);

        // We will use the InstructionOpcode enum to encode the opcodes
        // the appended digit to each opcode is the shift amount
        let inv_2 = AB::F::from_canonical_u32(2).inverse();
        let mut opcode_flags = vec![];
        for flag in flags {
            opcode_flags.push(flag * (flag - AB::F::ONE) * inv_2);
        }
        for flag in flags {
            opcode_flags.push(flag * (sum.clone() - AB::F::TWO) * AB::F::NEG_ONE);
        }
        (0..4).for_each(|i| {
            ((i + 1)..4).for_each(|j| opcode_flags.push(flags[i] * flags[j]));
        });

        let opcode_when = |idxs: &[InstructionOpcode]| -> AB::Expr {
            idxs.iter().fold(AB::Expr::ZERO, |acc, &idx| {
                acc + opcode_flags[idx as usize].clone()
            })
        };

        // Constrain that is_load matches the opcode
        builder.assert_eq(
            is_load,
            opcode_when(&[LoadW0, LoadHu0, LoadHu2, LoadBu0, LoadBu1, LoadBu2, LoadBu3]),
        );
        builder.when(is_load).assert_one(is_valid);

        // There are three parts to write_data:
        // - 1st limb is always read_data
        // - 2nd to (NUM_CELLS/2)th limbs are:
        //   - read_data if loadw/loadhu/storew/storeh
        //   - prev_data if storeb
        //   - zero if loadbu
        // - (NUM_CELLS/2 + 1)th to last limbs are:
        //   - read_data if loadw/storew
        //   - prev_data if storeb/storeh
        //   - zero if loadbu/loadhu
        // Shifting needs to be carefully handled in case by case basis
        // refer to [run_write_data] for the expected behavior in each case
        for (i, cell) in write_data.iter().enumerate() {
            // handling loads, expected_load_val = 0 if a store operation is happening
            let expected_load_val = if i == 0 {
                opcode_when(&[LoadW0, LoadHu0, LoadBu0]) * read_data[0]
                    + opcode_when(&[LoadBu1]) * read_data[1]
                    + opcode_when(&[LoadHu2, LoadBu2]) * read_data[2]
                    + opcode_when(&[LoadBu3]) * read_data[3]
            } else if i < NUM_CELLS / 2 {
                opcode_when(&[LoadW0, LoadHu0]) * read_data[i]
                    + opcode_when(&[LoadHu2]) * read_data[i + 2]
            } else {
                opcode_when(&[LoadW0]) * read_data[i]
            };

            // handling stores, expected_store_val = 0 if a load operation is happening
            let expected_store_val = if i == 0 {
                opcode_when(&[StoreW0, StoreH0, StoreB0]) * read_data[i]
                    + opcode_when(&[StoreH2, StoreB1, StoreB2, StoreB3]) * prev_data[i]
            } else if i == 1 {
                opcode_when(&[StoreB1]) * read_data[i - 1]
                    + opcode_when(&[StoreW0, StoreH0]) * read_data[i]
                    + opcode_when(&[StoreH2, StoreB0, StoreB2, StoreB3]) * prev_data[i]
            } else if i == 2 {
                opcode_when(&[StoreH2, StoreB2]) * read_data[i - 2]
                    + opcode_when(&[StoreW0]) * read_data[i]
                    + opcode_when(&[StoreH0, StoreB0, StoreB1, StoreB3]) * prev_data[i]
            } else if i == 3 {
                opcode_when(&[StoreB3]) * read_data[i - 3]
                    + opcode_when(&[StoreH2]) * read_data[i - 2]
                    + opcode_when(&[StoreW0]) * read_data[i]
                    + opcode_when(&[StoreH0, StoreB0, StoreB1, StoreB2]) * prev_data[i]
            } else {
                opcode_when(&[StoreW0]) * read_data[i]
                    + opcode_when(&[StoreB0, StoreB1, StoreB2, StoreB3]) * prev_data[i]
                    + opcode_when(&[StoreH0])
                        * if i < NUM_CELLS / 2 {
                            read_data[i]
                        } else {
                            prev_data[i]
                        }
                    + opcode_when(&[StoreH2])
                        * if i - 2 < NUM_CELLS / 2 {
                            read_data[i - 2]
                        } else {
                            prev_data[i]
                        }
            };
            let expected_val = expected_load_val + expected_store_val;
            builder.assert_eq(*cell, expected_val);
        }

        let expected_opcode = opcode_when(&[LoadW0]) * AB::Expr::from_canonical_u8(LOADW as u8)
            + opcode_when(&[LoadHu0, LoadHu2]) * AB::Expr::from_canonical_u8(LOADHU as u8)
            + opcode_when(&[LoadBu0, LoadBu1, LoadBu2, LoadBu3])
                * AB::Expr::from_canonical_u8(LOADBU as u8)
            + opcode_when(&[StoreW0]) * AB::Expr::from_canonical_u8(STOREW as u8)
            + opcode_when(&[StoreH0, StoreH2]) * AB::Expr::from_canonical_u8(STOREH as u8)
            + opcode_when(&[StoreB0, StoreB1, StoreB2, StoreB3])
                * AB::Expr::from_canonical_u8(STOREB as u8);
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);

        let load_shift_amount = opcode_when(&[LoadBu1]) * AB::Expr::ONE
            + opcode_when(&[LoadHu2, LoadBu2]) * AB::Expr::TWO
            + opcode_when(&[LoadBu3]) * AB::Expr::from_canonical_u32(3);

        let store_shift_amount = opcode_when(&[StoreB1]) * AB::Expr::ONE
            + opcode_when(&[StoreH2, StoreB2]) * AB::Expr::TWO
            + opcode_when(&[StoreB3]) * AB::Expr::from_canonical_u32(3);

        AdapterAirContext {
            to_pc: None,
            reads: (prev_data, read_data.map(|x| x.into())).into(),
            writes: [write_data.map(|x| x.into())].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                is_load: is_load.into(),
                load_shift_amount,
                store_shift_amount,
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
pub struct LoadStoreCoreRecord<const NUM_CELLS: usize> {
    pub local_opcode: u8,
    pub shift_amount: u8,
    pub read_data: [u8; NUM_CELLS],
    // Note: `prev_data` can be from native address space, so we need to use u32
    pub prev_data: [u32; NUM_CELLS],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadStoreStep<A, const NUM_CELLS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct LoadStoreFiller<
    A = Rv32LoadStoreAdapterFiller,
    const NUM_CELLS: usize = RV32_REGISTER_NUM_LIMBS,
> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const NUM_CELLS: usize> InstructionExecutor<F, RA> for LoadStoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceStep<
            F,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut LoadStoreCoreRecord<NUM_CELLS>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        (
            (core_record.prev_data, core_record.read_data),
            core_record.shift_amount,
        ) = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        core_record.local_opcode = local_opcode as u8;

        let write_data = run_write_data(
            local_opcode,
            core_record.read_data,
            core_record.prev_data,
            core_record.shift_amount as usize,
        );
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_CELLS: usize> TraceFiller<F> for LoadStoreFiller<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &LoadStoreCoreRecord<NUM_CELLS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut LoadStoreCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        let opcode = Rv32LoadStoreOpcode::from_usize(record.local_opcode as usize);
        let shift = record.shift_amount;

        let write_data = run_write_data(opcode, record.read_data, record.prev_data, shift as usize);
        // Writing in reverse order
        core_row.write_data = write_data.map(F::from_canonical_u32);
        core_row.prev_data = record.prev_data.map(F::from_canonical_u32);
        core_row.read_data = record.read_data.map(F::from_canonical_u8);
        core_row.is_load = F::from_bool([LOADW, LOADHU, LOADBU].contains(&opcode));
        core_row.is_valid = F::ONE;
        let flags = &mut core_row.flags;
        *flags = [F::ZERO; 4];
        match (opcode, shift) {
            (LOADW, 0) => flags[0] = F::TWO,
            (LOADHU, 0) => flags[1] = F::TWO,
            (LOADHU, 2) => flags[2] = F::TWO,
            (LOADBU, 0) => flags[3] = F::TWO,

            (LOADBU, 1) => flags[0] = F::ONE,
            (LOADBU, 2) => flags[1] = F::ONE,
            (LOADBU, 3) => flags[2] = F::ONE,
            (STOREW, 0) => flags[3] = F::ONE,

            (STOREH, 0) => (flags[0], flags[1]) = (F::ONE, F::ONE),
            (STOREH, 2) => (flags[0], flags[2]) = (F::ONE, F::ONE),
            (STOREB, 0) => (flags[0], flags[3]) = (F::ONE, F::ONE),
            (STOREB, 1) => (flags[1], flags[2]) = (F::ONE, F::ONE),
            (STOREB, 2) => (flags[1], flags[3]) = (F::ONE, F::ONE),
            (STOREB, 3) => (flags[2], flags[3]) = (F::ONE, F::ONE),
            _ => unreachable!(),
        };
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadStorePreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<F, A, const NUM_CELLS: usize> InsExecutorE1<F> for LoadStoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadStorePreCompute>()
    }

    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LoadStorePreCompute = data.borrow_mut();
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, pre_compute)?;
        let fn_ptr = match (local_opcode, enabled, is_native_store) {
            (LOADW, true, _) => execute_e1_impl::<_, _, U8, LoadWOp, true>,
            (LOADW, false, _) => execute_e1_impl::<_, _, U8, LoadWOp, false>,
            (LOADHU, true, _) => execute_e1_impl::<_, _, U8, LoadHUOp, true>,
            (LOADHU, false, _) => execute_e1_impl::<_, _, U8, LoadHUOp, false>,
            (LOADBU, true, _) => execute_e1_impl::<_, _, U8, LoadBUOp, true>,
            (LOADBU, false, _) => execute_e1_impl::<_, _, U8, LoadBUOp, false>,
            (STOREW, true, false) => execute_e1_impl::<_, _, U8, StoreWOp, true>,
            (STOREW, false, false) => execute_e1_impl::<_, _, U8, StoreWOp, false>,
            (STOREW, true, true) => execute_e1_impl::<_, _, F, StoreWOp, true>,
            (STOREW, false, true) => execute_e1_impl::<_, _, F, StoreWOp, false>,
            (STOREH, true, false) => execute_e1_impl::<_, _, U8, StoreHOp, true>,
            (STOREH, false, false) => execute_e1_impl::<_, _, U8, StoreHOp, false>,
            (STOREH, true, true) => execute_e1_impl::<_, _, F, StoreHOp, true>,
            (STOREH, false, true) => execute_e1_impl::<_, _, F, StoreHOp, false>,
            (STOREB, true, false) => execute_e1_impl::<_, _, U8, StoreBOp, true>,
            (STOREB, false, false) => execute_e1_impl::<_, _, U8, StoreBOp, false>,
            (STOREB, true, true) => execute_e1_impl::<_, _, F, StoreBOp, true>,
            (STOREB, false, true) => execute_e1_impl::<_, _, F, StoreBOp, false>,
            (_, _, _) => unreachable!(),
        };
        Ok(fn_ptr)
    }
}

impl<F, A, const NUM_CELLS: usize> InsExecutorE2<F> for LoadStoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
{
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LoadStorePreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E2ExecutionCtx,
    {
        let pre_compute: &mut E2PreCompute<LoadStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        let fn_ptr = match (local_opcode, enabled, is_native_store) {
            (LOADW, true, _) => execute_e2_impl::<_, _, U8, LoadWOp, true>,
            (LOADW, false, _) => execute_e2_impl::<_, _, U8, LoadWOp, false>,
            (LOADHU, true, _) => execute_e2_impl::<_, _, U8, LoadHUOp, true>,
            (LOADHU, false, _) => execute_e2_impl::<_, _, U8, LoadHUOp, false>,
            (LOADBU, true, _) => execute_e2_impl::<_, _, U8, LoadBUOp, true>,
            (LOADBU, false, _) => execute_e2_impl::<_, _, U8, LoadBUOp, false>,
            (STOREW, true, false) => execute_e2_impl::<_, _, U8, StoreWOp, true>,
            (STOREW, false, false) => execute_e2_impl::<_, _, U8, StoreWOp, false>,
            (STOREW, true, true) => execute_e2_impl::<_, _, F, StoreWOp, true>,
            (STOREW, false, true) => execute_e2_impl::<_, _, F, StoreWOp, false>,
            (STOREH, true, false) => execute_e2_impl::<_, _, U8, StoreHOp, true>,
            (STOREH, false, false) => execute_e2_impl::<_, _, U8, StoreHOp, false>,
            (STOREH, true, true) => execute_e2_impl::<_, _, F, StoreHOp, true>,
            (STOREH, false, true) => execute_e2_impl::<_, _, F, StoreHOp, false>,
            (STOREB, true, false) => execute_e2_impl::<_, _, U8, StoreBOp, true>,
            (STOREB, false, false) => execute_e2_impl::<_, _, U8, StoreBOp, false>,
            (STOREB, true, true) => execute_e2_impl::<_, _, F, StoreBOp, true>,
            (STOREB, false, true) => execute_e2_impl::<_, _, F, StoreBOp, false>,
            (_, _, _) => unreachable!(),
        };
        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    T: Copy + Debug + Default,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &LoadStorePreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = u32::from_le_bytes(rs1_bytes);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    // sign_extend([r32{c,g}(b):2]_e)`
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));
    let shift_amount = ptr_val % 4;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = if OP::IS_LOAD {
        vm_state.vm_read(pre_compute.e as u32, ptr_val)
    } else {
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32)
    };

    // We need to write 4 u32s for STORE.
    let mut write_data: [T; RV32_REGISTER_NUM_LIMBS] = if OP::HOST_READ {
        vm_state.host_read(pre_compute.e as u32, ptr_val)
    } else {
        [T::default(); RV32_REGISTER_NUM_LIMBS]
    };

    if !OP::compute_write_data(&mut write_data, read_data, shift_amount as usize) {
        vm_state.exit_code = Err(ExecutionError::Fail { pc: vm_state.pc });
        return;
    }

    if ENABLED {
        if OP::IS_LOAD {
            vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &write_data);
        } else {
            vm_state.vm_write(pre_compute.e as u32, ptr_val, &write_data);
        }
    }

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    T: Copy + Debug + Default,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &LoadStorePreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    T: Copy + Debug + Default,
    OP: LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<LoadStorePreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(&pre_compute.data, vm_state);
}

impl<A, const NUM_CELLS: usize> LoadStoreStep<A, NUM_CELLS> {
    /// Return (local_opcode, enabled, is_native_store)
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadStorePreCompute,
    ) -> Result<(Rv32LoadStoreOpcode, bool, bool), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;
        let enabled = !f.is_zero();

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            LOADW | LOADBU | LOADHU => {}
            STOREW | STOREH | STOREB => {
                if !enabled {
                    return Err(StaticProgramError::InvalidInstruction(pc));
                }
            }
            _ => unreachable!("LoadStoreStep should not handle LOADB/LOADH opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;
        let is_native_store = e_u32 == NATIVE_AS;

        *data = LoadStorePreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        Ok((local_opcode, enabled, is_native_store))
    }
}

trait LoadStoreOp<T> {
    const IS_LOAD: bool;
    const HOST_READ: bool;

    /// Return if the operation is valid.
    fn compute_write_data(
        write_data: &mut [T; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool;
}
/// Wrapper type for u8 so we can implement `LoadStoreOp<F>` for `F: PrimeField32`.
/// For memory read/write, this type behaves as same as `u8`.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default)]
struct U8(u8);
struct LoadWOp;
struct LoadHUOp;
struct LoadBUOp;
struct StoreWOp;
struct StoreHOp;
struct StoreBOp;
impl LoadStoreOp<U8> for LoadWOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}

impl LoadStoreOp<U8> for LoadHUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 {
            return false;
        }
        write_data[0] = U8(read_data[shift_amount]);
        write_data[1] = U8(read_data[shift_amount + 1]);
        true
    }
}
impl LoadStoreOp<U8> for LoadBUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[0] = U8(read_data[shift_amount]);
        true
    }
}

impl LoadStoreOp<U8> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}
impl LoadStoreOp<U8> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 {
            return false;
        }
        write_data[shift_amount] = U8(read_data[0]);
        write_data[shift_amount + 1] = U8(read_data[1]);
        true
    }
}
impl LoadStoreOp<U8> for StoreBOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = U8(read_data[0]);
        true
    }
}

impl<F: PrimeField32> LoadStoreOp<F> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(F::from_canonical_u8);
        true
    }
}
impl<F: PrimeField32> LoadStoreOp<F> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 2 {
            return false;
        }
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        write_data[shift_amount + 1] = F::from_canonical_u8(read_data[1]);
        true
    }
}
impl<F: PrimeField32> LoadStoreOp<F> for StoreBOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;
    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV32_REGISTER_NUM_LIMBS],
        read_data: [u8; RV32_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        true
    }
}

// Returns the write data
#[inline(always)]
pub(super) fn run_write_data<const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    prev_data: [u32; NUM_CELLS],
    shift: usize,
) -> [u32; NUM_CELLS] {
    match (opcode, shift) {
        (LOADW, 0) => {
            read_data.map(|x| x as u32)
        },
        (LOADBU, 0) | (LOADBU, 1) | (LOADBU, 2) | (LOADBU, 3) => {
           let mut wrie_data = [0; NUM_CELLS];
           wrie_data[0] = read_data[shift] as u32;
           wrie_data
        }
        (LOADHU, 0) | (LOADHU, 2) => {
            let mut write_data = [0; NUM_CELLS];
            for (i, cell) in write_data.iter_mut().take(NUM_CELLS / 2).enumerate() {
                *cell = read_data[i + shift] as u32;
            }
            write_data
        }
        (STOREW, 0) => {
            read_data.map(|x| x as u32)
        },
        (STOREB, 0) | (STOREB, 1) | (STOREB, 2) | (STOREB, 3) => {
            let mut write_data = prev_data;
            write_data[shift] = read_data[0] as u32;
            write_data
        }
        (STOREH, 0) | (STOREH, 2) => {
            array::from_fn(|i| {
                if i >= shift && i < (NUM_CELLS / 2 + shift){
                    read_data[i - shift] as u32
                } else {
                    prev_data[i]
                }
            })
        }
        // Currently the adapter AIR requires `ptr_val` to be aligned to the data size in bytes.
        // The circuit requires that `shift = ptr_val % 4` so that `ptr_val - shift` is a multiple of 4.
        // This requirement is non-trivial to remove, because we use it to ensure that `ptr_val - shift + 4 <= 2^pointer_max_bits`.
        _ => unreachable!(
            "unaligned memory access not supported by this execution environment: {opcode:?}, shift: {shift}"
        ),
    }
}
