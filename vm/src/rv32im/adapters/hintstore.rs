use std::{
    array,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use ax_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::instruction::Instruction;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{compose, RV32_REGISTER_NUM_LIMBS};
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    rv32im::adapters::RV32_CELL_BITS,
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};

/// This chip reads rs1 and gets a intermediate memory pointer address with rs1 + imm.
/// It writes to the memory at the intermediate pointer.
#[derive(Debug)]
pub struct Rv32HintStoreAdapterChip<F: Field> {
    pub air: Rv32HintStoreAdapterAir,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Rv32HintStoreAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        range_checker_chip: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: Rv32HintStoreAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                range_bus: range_checker_chip.bus(),
                pointer_max_bits: memory_controller.mem_config.pointer_max_bits,
            },
            range_checker_chip,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32HintStoreReadRecord<F: Field> {
    pub rs1_record: MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>,
    pub rs1_ptr: F,

    pub imm: F,
    pub imm_sign: bool,
    pub mem_ptr_limbs: [F; 2],
}

#[derive(Debug, Clone)]
pub struct Rv32HintStoreWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub write: MemoryWriteRecord<F, RV32_REGISTER_NUM_LIMBS>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32HintStoreAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>,

    pub imm: T,
    pub imm_sign: T,
    /// mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    pub mem_ptr_limbs: [T; 2],
    pub write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32HintStoreAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv32HintStoreAdapterAir {
    fn width(&self) -> usize {
        Rv32HintStoreAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32HintStoreAdapterAir {
    /// The HintStoreAdapter handles memory writes and getting the intermediate memory pointer.
    /// This chip handles the HintStoreW instruction, so it doesn't constrain the data read from memory.
    ///
    /// Here 1 write represents the data that needs to be written to memory
    /// Getting the intermediate pointer is completely internal to the adapter
    /// and shouldn't be a part of the AdapterInterface
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        0,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32HintStoreAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let is_valid = ctx.instruction.is_valid;

        // read rs1
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local_cols.rs1_ptr),
                local_cols.rs1_data,
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, is_valid.clone());

        // constrain mem_ptr = rs1 + imm as a u32 addition with 2 limbs
        let limbs_01 = local_cols.rs1_data[0]
            + local_cols.rs1_data[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let limbs_23 = local_cols.rs1_data[2]
            + local_cols.rs1_data[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);

        let inv = AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2)).inverse();
        let carry = (limbs_01 + local_cols.imm - local_cols.mem_ptr_limbs[0]) * inv;

        builder.assert_bool(carry.clone());

        builder.assert_bool(local_cols.imm_sign);
        let imm_extend_limb =
            local_cols.imm_sign * AB::F::from_canonical_u32((1 << (RV32_CELL_BITS * 2)) - 1);
        let carry = (limbs_23 + imm_extend_limb + carry - local_cols.mem_ptr_limbs[1]) * inv;
        builder.assert_bool(carry.clone());

        // preventing mem_ptr overflow
        self.range_bus
            .range_check(local_cols.mem_ptr_limbs[0], RV32_CELL_BITS * 2)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(
                local_cols.mem_ptr_limbs[1],
                self.pointer_max_bits - RV32_CELL_BITS * 2,
            )
            .eval(builder, is_valid.clone());

        let mem_ptr = local_cols.mem_ptr_limbs[0]
            + local_cols.mem_ptr_limbs[1] * AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2));

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::two(), mem_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, is_valid.clone());

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(4));
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    AB::Expr::zero(),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    AB::Expr::one(),
                    AB::Expr::two(),
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let local_cols: &Rv32HintStoreAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32HintStoreAdapterChip<F> {
    type ReadRecord = Rv32HintStoreReadRecord<F>;
    type WriteRecord = Rv32HintStoreWriteRecord<F>;
    type Air = Rv32HintStoreAdapterAir;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        0,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, d, e, .. } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);
        assert!(self.range_checker_chip.range_max_bits() >= 16);

        let rs1_record = memory.read::<RV32_REGISTER_NUM_LIMBS>(d, b);
        let rs1_val = compose(rs1_record.data);
        let imm = c.as_canonical_u32();
        let imm_sign = (imm & 0x8000) >> 15;
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        assert!(ptr_val < (1 << self.air.pointer_max_bits));
        let mem_ptr_limbs = array::from_fn(|i| ((ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff));
        self.range_checker_chip
            .add_count(mem_ptr_limbs[0], RV32_CELL_BITS * 2);
        self.range_checker_chip.add_count(
            mem_ptr_limbs[1],
            self.air.pointer_max_bits - RV32_CELL_BITS * 2,
        );

        Ok((
            [],
            Self::ReadRecord {
                rs1_record,
                rs1_ptr: b,
                imm: c,
                imm_sign: imm_sign == 1,
                mem_ptr_limbs: mem_ptr_limbs.map(F::from_canonical_u32),
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let ptr = read_record.mem_ptr_limbs[0]
            + read_record.mem_ptr_limbs[1] * F::from_canonical_u32(1 << (RV32_CELL_BITS * 2));
        let write_record = memory.write(instruction.e, ptr, output.writes[0]);

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state,
                write: write_record,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let adapter_cols: &mut Rv32HintStoreAdapterCols<_> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.rs1_data = read_record.rs1_record.data;
        adapter_cols.rs1_aux_cols = aux_cols_factory.make_read_aux_cols(read_record.rs1_record);
        adapter_cols.rs1_ptr = read_record.rs1_ptr;
        adapter_cols.imm = read_record.imm;
        adapter_cols.imm_sign = F::from_bool(read_record.imm_sign);
        adapter_cols.mem_ptr_limbs = read_record.mem_ptr_limbs;
        adapter_cols.write_aux = aux_cols_factory.make_write_aux_cols(write_record.write);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
