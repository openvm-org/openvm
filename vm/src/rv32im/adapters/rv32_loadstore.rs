use std::{
    array,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    utils,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::{compose, ADDRESS_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    rv32im::adapters::RV32_CELL_BITS,
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

pub struct LoadStoreProcessedInstruction<T> {
    pub is_valid: T,
    // Absolute opcode number
    pub opcode: T,
    pub is_load: T,
    pub is_hint: T,
}

// The LoadStoreAdapter seperates Runtime and Air AdapterInterfaces.
// This is necessary because `prev_data` should be owned by the core chip and sent to the adapter,
// and it must have an AB::Var type in AIR as to satisfy the memory_bridge interface.
// This is achived by having different types for reads and writes in Air AdapterInterface.
// This method ensures that there are no modifications to the global interfaces.

// Here 2 reads represent read_data and prev_data,
// Getting the intermediate pointer is completely internal to the adapter and shouldn't be a part of the AdapterInterface
type Rv32LoadStoreAdapterRuntimeInterface<T> = BasicAdapterInterface<
    T,
    LoadStoreProcessedInstruction<T>,
    2,
    1,
    RV32_REGISTER_NUM_LIMBS,
    RV32_REGISTER_NUM_LIMBS,
>;

pub struct Rv32LoadStoreAdapterAirInterface<AB: InteractionBuilder>(PhantomData<AB>);

impl<AB: InteractionBuilder> VmAdapterInterface<AB::Expr> for Rv32LoadStoreAdapterAirInterface<AB> {
    type Reads = [[AB::Var; RV32_REGISTER_NUM_LIMBS]; 2];
    type Writes = [[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1];
    type ProcessedInstruction = LoadStoreProcessedInstruction<AB::Expr>;
}

// This chip reads rs1 and gets a intermediate memory pointer address with rs1 + imm.
// In case of Loads, reads from the intermediate pointer and writes to rd.
// In case of Stores, reads from rs2 and writes to the intermediate pointer.
#[derive(Debug, Clone)]
pub struct Rv32LoadStoreAdapterChip<F: Field> {
    pub air: Rv32LoadStoreAdapterAir,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    offset: usize,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Rv32LoadStoreAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        range_checker_chip: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: Rv32LoadStoreAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                range_bus: range_checker_chip.bus(),
            },
            range_checker_chip,
            offset,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreReadRecord<F: Field> {
    pub rs1_record: Option<MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>>,
    pub rs1_ptr: F,
    // This will be a read from a register in case of Stores and a read from RISC-V memory in case of Loads
    pub read: Option<MemoryReadRecord<F, RV32_REGISTER_NUM_LIMBS>>,

    pub imm: F,
    pub imm_sign: bool,
    pub mem_ptr: [F; 2],
}

#[derive(Debug, Clone)]
pub struct Rv32LoadStoreWriteRecord<F: Field> {
    pub from_state: ExecutionState<F>,
    // This will be a write to a register in case of Load and a write to RISC-V memory in case of Stores
    pub write: MemoryWriteRecord<F, RV32_REGISTER_NUM_LIMBS>,
    pub rd_rs2_ptr: F,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32LoadStoreAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>,

    // Will write to rd when Load and read from rs2 when Store
    pub rd_rs2_ptr: T,
    pub read_data_aux: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    pub imm: T,
    pub imm_sign: T,
    // mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    pub mem_ptr: [T; 2],

    // prev_data will be provided by the core chip to make a complete MemoryWriteAuxCols
    pub write_aux: [T; MemoryWriteAuxCols::<u8, RV32_REGISTER_NUM_LIMBS>::striped_width()],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32LoadStoreAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv32LoadStoreAdapterAir {
    fn width(&self) -> usize {
        Rv32LoadStoreAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32LoadStoreAdapterAir {
    type Interface = Rv32LoadStoreAdapterAirInterface<AB>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32LoadStoreAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let is_load = ctx.instruction.is_load;
        let is_valid = ctx.instruction.is_valid;
        let is_hint = ctx.instruction.is_hint;

        // read rs1
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::one(), local_cols.rs1_ptr),
                local_cols.rs1_data,
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, is_valid.clone() - is_hint.clone());

        // constrain mem_ptr = rs1 + imm as a u32 addition with 2 limbs
        let limbs_01 = local_cols.rs1_data[0]
            + local_cols.rs1_data[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let limbs_23 = local_cols.rs1_data[2]
            + local_cols.rs1_data[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);

        let inv = AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2)).inverse();
        let carry = (limbs_01 + local_cols.imm - local_cols.mem_ptr[0]) * inv;

        builder
            .when(is_valid.clone() - is_hint.clone())
            .assert_bool(carry.clone());

        builder
            .when(is_valid.clone() - is_hint.clone())
            .assert_bool(local_cols.imm_sign);
        let imm_extend_limb =
            local_cols.imm_sign * AB::F::from_canonical_u32((1 << (RV32_CELL_BITS * 2)) - 1);
        let carry = (limbs_23 + imm_extend_limb + carry - local_cols.mem_ptr[1]) * inv;
        builder
            .when(is_valid.clone() - is_hint.clone())
            .assert_bool(carry.clone());

        // preventing mem_ptr overflow
        self.range_bus
            .range_check(local_cols.mem_ptr[0], RV32_CELL_BITS * 2)
            .eval(builder, is_valid.clone() - is_hint.clone());
        self.range_bus
            .range_check(local_cols.mem_ptr[1], ADDRESS_BITS - RV32_CELL_BITS * 2)
            .eval(builder, is_valid.clone() - is_hint.clone());

        let mem_ptr = local_cols.mem_ptr[0]
            + local_cols.mem_ptr[1] * AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2));

        // read_as is 2 for loads and 1 for stores
        let read_as =
            is_load.clone() * AB::F::two() + utils::not::<AB::Expr>(is_load.clone()) * AB::F::one();

        // read_ptr is mem_ptr for loads and rd_rs2_ptr for stores
        let read_ptr = is_load.clone() * mem_ptr.clone()
            + utils::not::<AB::Expr>(is_load.clone()) * local_cols.rd_rs2_ptr;

        self.memory_bridge
            .read(
                MemoryAddress::new(read_as, read_ptr),
                ctx.reads[1],
                timestamp_pp(),
                &local_cols.read_data_aux,
            )
            .eval(builder, is_valid.clone() - is_hint.clone());

        let write_aux_cols =
            MemoryWriteAuxCols::concat_prev_data(&local_cols.write_aux, ctx.reads[0]);

        // write_as is 1 for loads and 2 for stores
        let write_as =
            is_load.clone() * AB::F::one() + utils::not::<AB::Expr>(is_load.clone()) * AB::F::two();

        // write_ptr is rd_rs2_ptr for loads and mem_ptr for stores
        let write_ptr = is_load.clone() * local_cols.rd_rs2_ptr
            + utils::not::<AB::Expr>(is_load.clone()) * mem_ptr;

        self.memory_bridge
            .write(
                MemoryAddress::new(write_as, write_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &write_aux_cols,
            )
            .eval(builder, is_valid.clone());

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(4));
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_rs2_ptr.into(),
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
        let local_cols: &Rv32LoadStoreAdapterCols<AB::Var> = local.borrow();
        local_cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32LoadStoreAdapterChip<F> {
    type ReadRecord = Rv32LoadStoreReadRecord<F>;
    type WriteRecord = Rv32LoadStoreWriteRecord<F>;
    type Air = Rv32LoadStoreAdapterAir;
    type Interface = Rv32LoadStoreAdapterRuntimeInterface<F>;

    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), 1);
        debug_assert_eq!(e.as_canonical_u32(), 2);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(opcode - self.offset);
        let rs1_record = if local_opcode == HINTLOAD_RV32 {
            memory.increment_timestamp();
            None
        } else {
            Some(memory.read::<RV32_REGISTER_NUM_LIMBS>(d, b))
        };

        let rs1_val = match rs1_record {
            Some(rs1_record) => compose(rs1_record.data),
            None => 0,
        };
        let imm = c.as_canonical_u32();
        let imm_sign = (imm & 0x8000) >> 15;
        let imm_extended = imm + imm_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(imm_extended);
        assert!(ptr_val < (1 << ADDRESS_BITS) || local_opcode == HINTLOAD_RV32);
        let mem_ptr = array::from_fn(|i| (ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff);

        if local_opcode != HINTLOAD_RV32 {
            self.range_checker_chip
                .add_count(mem_ptr[0], RV32_CELL_BITS * 2);
            self.range_checker_chip
                .add_count(mem_ptr[1], ADDRESS_BITS - RV32_CELL_BITS * 2);
        }
        let read_record = match local_opcode {
            LOADW | LOADB | LOADH | LOADBU | LOADHU => {
                Some(memory.read::<RV32_REGISTER_NUM_LIMBS>(e, F::from_canonical_u32(ptr_val)))
            }
            STOREW | STOREH | STOREB => Some(memory.read::<RV32_REGISTER_NUM_LIMBS>(d, a)),
            HINTLOAD_RV32 => {
                memory.increment_timestamp();
                None
            }
        };

        // We need to keep values of some cells to keep them unchanged when writing to those cells
        let prev_data = match local_opcode {
            STOREW | STOREH | STOREB => array::from_fn(|i| {
                memory.unsafe_read_cell(e, F::from_canonical_usize(ptr_val as usize + i))
            }),
            LOADW | LOADB | LOADH | LOADBU | LOADHU | HINTLOAD_RV32 => {
                array::from_fn(|i| memory.unsafe_read_cell(d, a + F::from_canonical_usize(i)))
            }
        };

        let read_data = if let Some(read_record) = read_record {
            read_record.data
        } else {
            [F::zero(); RV32_REGISTER_NUM_LIMBS]
        };

        Ok((
            [prev_data, read_data],
            Self::ReadRecord {
                rs1_record,
                rs1_ptr: b,
                read: read_record,
                imm: c,
                imm_sign: imm_sign == 1,
                mem_ptr: mem_ptr.map(F::from_canonical_u32),
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
        let Instruction {
            opcode, a, d, e, ..
        } = *instruction;

        let local_opcode = Rv32LoadStoreOpcode::from_usize(opcode - self.offset);

        let rs1_data = match read_record.rs1_record {
            Some(rs1_record) => rs1_record.data,
            None => [F::zero(); RV32_REGISTER_NUM_LIMBS],
        };
        let write_record = match local_opcode {
            STOREW | STOREH | STOREB => {
                let rs1_val = compose(rs1_data);
                let imm = read_record.imm.as_canonical_u32();
                let imm_sign = read_record.imm_sign as u32;
                let imm_extended = imm + imm_sign * 0xffff0000;
                let ptr = rs1_val.wrapping_add(imm_extended);
                assert!(ptr < (1 << ADDRESS_BITS));
                memory.write(e, F::from_canonical_u32(ptr), output.writes[0])
            }
            LOADW | LOADB | LOADH | LOADBU | LOADHU | HINTLOAD_RV32 => {
                memory.write(d, a, output.writes[0])
            }
        };

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state: from_state.map(F::from_canonical_u32),
                write: write_record,
                rd_rs2_ptr: a,
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
        let adapter_cols: &mut Rv32LoadStoreAdapterCols<_> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state;
        (adapter_cols.rs1_data, adapter_cols.rs1_aux_cols) = match read_record.rs1_record {
            Some(rs1_record) => (
                rs1_record.data,
                aux_cols_factory.make_read_aux_cols(rs1_record),
            ),
            None => (
                [F::zero(); RV32_REGISTER_NUM_LIMBS],
                MemoryReadAuxCols::disabled(),
            ),
        };
        adapter_cols.rs1_ptr = read_record.rs1_ptr;
        adapter_cols.rd_rs2_ptr = write_record.rd_rs2_ptr;
        adapter_cols.read_data_aux = match read_record.read {
            Some(read) => aux_cols_factory.make_read_aux_cols(read),
            None => MemoryReadAuxCols::disabled(),
        };
        adapter_cols.imm = read_record.imm;
        adapter_cols.imm_sign = F::from_bool(read_record.imm_sign);
        adapter_cols.mem_ptr = read_record.mem_ptr;
        aux_cols_factory
            .make_write_aux_cols(write_record.write)
            .strip_prev_data(&mut adapter_cols.write_aux);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
