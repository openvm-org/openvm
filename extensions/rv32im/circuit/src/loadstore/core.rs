use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, Result, StepExecutorE1, TraceStep,
        VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{memory_read, memory_read_from_state, LoadStoreInstruction};

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

use openvm_circuit::{
    arch::{ExecuteFunc, PreComputeInstruction, VmSegmentState},
    next_instruction,
};
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_IMM_AS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use InstructionOpcode::*;
use openvm_circuit::arch::{get_record_from_slice, AdapterTraceFiller, EmptyAdapterCoreLayout, RecordArena, TraceFiller};

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

#[derive(derive_new::new)]
pub struct LoadStoreStep<A, const NUM_CELLS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, CTX, A, const NUM_CELLS: usize> TraceStep<F, CTX> for LoadStoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
{
    type RecordLayout = EmptyAdapterCoreLayout<F, A>;
    type RecordMut<'a> = (A::RecordMut<'a>, &'a mut LoadStoreCoreRecord<NUM_CELLS>);

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

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

impl<F, CTX, A, const NUM_CELLS: usize> TraceFiller<F, CTX> for LoadStoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F, CTX>,
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
    local_opcode: Rv32LoadStoreOpcode,
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
    enabled: bool,
}

impl<F, A, const NUM_CELLS: usize> StepExecutorE1<F> for LoadStoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn execute_e1<Ctx>(&self) -> ExecuteFunc<F, Ctx>
    where
        Ctx: E1E2ExecutionCtx,
    {
        execute_e1_impl
    }

    // fn execute_metered(
    //     &self,
    //     state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
    //     instruction: &Instruction<F>,
    //     chip_index: usize,
    // ) -> Result<()> {
    //     self.execute_e1(state, instruction)?;
    //     state.ctx.trace_heights[chip_index] += 1;
    //
    //     Ok(())
    // }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadStorePreCompute>()
    }

    #[inline(always)]
    fn pre_compute(&self, inst: &Instruction<F>, data: &mut [u8]) {
        let pre_compute: &mut LoadStorePreCompute = data.borrow_mut();
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
        assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        assert_ne!(e_u32, RV32_IMM_AS);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            LOADW | LOADBU | LOADHU => {}
            STOREW | STOREH | STOREB => {
                assert!(enabled)
            }
            _ => unreachable!("LoadStoreStep should not handle LOADB/LOADH opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        *pre_compute = LoadStorePreCompute {
            local_opcode,
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
            enabled,
        };
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let next_inst = unsafe { inst.offset(1) };
    let curr_inst = unsafe { &*inst };
    let pre_compute: &LoadStorePreCompute = curr_inst.pre_compute.borrow();

    let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
        vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = u32::from_le_bytes(rs1_bytes);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    // sign_extend([r32{c,g}(b):2]_e)`
    assert!(ptr_val < (1 << 29));
    let shift_amount = ptr_val % 4;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = match pre_compute.local_opcode {
        LOADW | LOADBU | LOADHU => vm_state.vm_read(pre_compute.e as u32, ptr_val),
        STOREW | STOREH | STOREB => vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32),
        _ => unreachable!("LoadStoreStep should not handle LOADB/LOADH opcodes"),
    };

    // We need to write 4-byte for STORE.
    let mut write_data: [u8; RV32_REGISTER_NUM_LIMBS] = match pre_compute.local_opcode {
        STOREH | STOREB => vm_state.host_read(pre_compute.e as u32, ptr_val),
        _ => [0; RV32_REGISTER_NUM_LIMBS],
    };

    match (pre_compute.local_opcode, shift_amount) {
        (LOADW, 0) | (STOREW, 0) => write_data = read_data,
        (LOADBU, 0) | (LOADBU, 1) | (LOADBU, 2) | (LOADBU, 3) => {
            write_data[0] = read_data[shift_amount as usize];
        }
        (STOREB, 0) | (STOREB, 1) | (STOREB, 2) | (STOREB, 3) => {
            write_data[shift_amount as usize] = read_data[0];
        }
        (LOADHU, 0) | (LOADHU, 2) => {
            write_data[0] = read_data[shift_amount as usize];
            write_data[1] = read_data[shift_amount as usize + 1];
        }
        (STOREH, 0) | (STOREH, 2) => {
            write_data[shift_amount as usize] = read_data[0];
            write_data[shift_amount as usize + 1] = read_data[1];
        }
        _ => unreachable!("LoadStoreStep should not handle LOADB/LOADH opcodes"),
    };

    if pre_compute.enabled {
        match pre_compute.local_opcode {
            STOREW | STOREH | STOREB => {
                vm_state.vm_write(pre_compute.e as u32, ptr_val, &write_data);
            }
            LOADW | LOADBU | LOADHU => {
                vm_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &write_data);
            }
            _ => unreachable!("LoadStoreStep should not handle LOADB/LOADH opcodes"),
        }
    }

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;

    next_instruction!(next_inst, vm_state)
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
