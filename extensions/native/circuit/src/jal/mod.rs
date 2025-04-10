use std::{
    borrow::{Borrow, BorrowMut},
    ops::Deref,
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionError, ExecutionState, InstructionExecutor, PcIncOrSet},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory, MemoryController, OfflineMemory, RecordId,
    },
};
use openvm_circuit_primitives::{
    utils::next_power_of_two_or_zero,
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, NativeJalOpcode, NativeRangeCheckOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;
use AS::Native;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(AlignedBorrow)]
struct JalRangeCheckCols<T> {
    is_jal: T,
    is_range_check: T,
    a_pointer: T,
    state: ExecutionState<T>,
    // Write when is_jal, read when is_range_check.
    writes_aux: MemoryWriteAuxCols<T, 1>,
    b: T,
    // Only used by range check.
    c: T,
    // Only used by range check.
    y: T,
}

const OVERALL_WIDTH: usize = JalRangeCheckCols::<u8>::width();
const_assert_eq!(OVERALL_WIDTH, 12);

#[derive(Copy, Clone, Debug)]
pub struct JalRangeCheckAir {
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for JalRangeCheckAir {
    fn width(&self) -> usize {
        OVERALL_WIDTH
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JalRangeCheckAir {}
impl<F: Field> PartitionedBaseAir<F> for JalRangeCheckAir {}
impl<AB: InteractionBuilder> Air<AB> for JalRangeCheckAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_slice = local.deref();
        let local: &JalRangeCheckCols<AB::Var> = local_slice.borrow();
        builder.assert_bool(local.is_jal);
        builder.assert_bool(local.is_range_check);
        let is_valid = local.is_jal + local.is_range_check;
        builder.assert_bool(is_valid.clone());

        let d = AB::Expr::from_canonical_u32(Native as u32);
        let a_val = local.writes_aux.prev_data()[0];
        // if is_jal, write pc + DEFAULT_PC_STEP, else if is_range_check, read a_val.
        let write_val = local.is_jal
            * (local.state.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_STEP))
            + local.is_range_check * a_val;
        self.memory_bridge
            .write(
                MemoryAddress::new(d.clone(), local.a_pointer),
                [write_val],
                local.state.timestamp,
                &local.writes_aux,
            )
            .eval(builder, is_valid.clone());

        let opcode = local.is_jal
            * AB::F::from_canonical_usize(NativeJalOpcode::JAL.global_opcode().as_usize())
            + local.is_range_check
                * AB::F::from_canonical_usize(
                    NativeRangeCheckOpcode::RANGE_CHECK
                        .global_opcode()
                        .as_usize(),
                );
        // Increment pc by b if is_jal, else by DEFAULT_PC_STEP if is_range_check.
        let pc_inc = local.is_jal * local.b
            + local.is_range_check * AB::F::from_canonical_u32(DEFAULT_PC_STEP);
        builder.when(local.is_jal).assert_zero(local.c);
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                opcode,
                [local.a_pointer.into(), local.b.into(), local.c.into(), d],
                local.state,
                AB::F::ONE,
                PcIncOrSet::Inc(pc_inc),
            )
            .eval(builder, is_valid);

        // Range check specific:
        // a_val = x + y * (1 << 16)
        let x = a_val - local.y * AB::Expr::from_canonical_u32(1 << 16);
        self.range_bus
            .send(x.clone(), local.b)
            .eval(builder, local.is_range_check);
        // Assert y < (1 << c), where c <= 14.
        self.range_bus
            .send(local.y, local.c)
            .eval(builder, local.is_range_check);
    }
}

impl JalRangeCheckAir {
    fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        Self {
            execution_bridge,
            memory_bridge,
            range_bus,
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct JalRangeCheckRecord {
    pub state: ExecutionState<u32>,
    pub a_rw: RecordId,
    pub b: u32,
    pub c: u8,
    pub is_jal: bool,
}

/// Chip for JAL and RANGE_CHECK. These opcodes are logically irrelevant. Putting these opcodes into
/// the same chip is just to save columns.
pub struct JalRangeCheckChip<F> {
    air: JalRangeCheckAir,
    pub records: Vec<JalRangeCheckRecord>,
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    range_checker_chip: SharedVariableRangeCheckerChip,
    /// If true, ignore execution errors.
    debug: bool,
}

impl<F: PrimeField32> JalRangeCheckChip<F> {
    pub fn new(
        execution_bridge: ExecutionBridge,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        let memory_bridge = offline_memory.lock().unwrap().memory_bridge();
        let air = JalRangeCheckAir::new(execution_bridge, memory_bridge, range_checker_chip.bus());
        Self {
            air,
            records: vec![],
            offline_memory,
            range_checker_chip,
            debug: false,
        }
    }
    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for JalRangeCheckChip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        if instruction.opcode == NativeJalOpcode::JAL.global_opcode() {
            let (record_id, _) = memory.write(
                F::from_canonical_u32(AS::Native as u32),
                instruction.a,
                &[F::from_canonical_u32(from_state.pc + DEFAULT_PC_STEP)],
            );
            let b = instruction.b.as_canonical_u32();
            self.records.push(JalRangeCheckRecord {
                state: from_state,
                a_rw: record_id,
                b,
                c: 0,
                is_jal: true,
            });
            return Ok(ExecutionState {
                pc: (F::from_canonical_u32(from_state.pc) + instruction.b).as_canonical_u32(),
                timestamp: memory.timestamp(),
            });
        } else if instruction.opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode() {
            let d = F::from_canonical_u32(AS::Native as u32);
            // This is a read, but we make the record have prev_data
            let a_val = memory.unsafe_read_cell::<F>(d, instruction.a);
            let (record_id, _) = memory.write(d, instruction.a, &[a_val]);
            let a_val = a_val.as_canonical_u32();
            let b = instruction.b.as_canonical_u32();
            let c = instruction.c.as_canonical_u32();
            debug_assert!(!self.debug || b <= 16);
            debug_assert!(!self.debug || c <= 14);
            let x = a_val & ((1 << 16) - 1);
            if !self.debug && x >= 1 << b {
                return Err(ExecutionError::Fail { pc: from_state.pc });
            }
            let y = a_val >> 16;
            if !self.debug && y >= 1 << c {
                return Err(ExecutionError::Fail { pc: from_state.pc });
            }
            self.records.push(JalRangeCheckRecord {
                state: from_state,
                a_rw: record_id,
                b,
                c: c as u8,
                is_jal: false,
            });
            return Ok(ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            });
        }
        panic!("Unknown opcode {}", instruction.opcode);
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let jal_opcode = NativeJalOpcode::JAL.global_opcode().as_usize();
        let range_check_opcode = NativeRangeCheckOpcode::RANGE_CHECK
            .global_opcode()
            .as_usize();
        if opcode == jal_opcode {
            return String::from("JAL");
        }
        if opcode == range_check_opcode {
            return String::from("RANGE_CHECK");
        }
        panic!("Unknown opcode {}", opcode);
    }
}

impl<F: Field> ChipUsageGetter for JalRangeCheckChip<F> {
    fn air_name(&self) -> String {
        "JalRangeCheck".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        OVERALL_WIDTH
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for JalRangeCheckChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }
    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let height = next_power_of_two_or_zero(self.records.len());
        let mut flat_trace = Val::<SC>::zero_vec(OVERALL_WIDTH * height);
        let memory = self.offline_memory.lock().unwrap();
        let aux_cols_factory = memory.aux_cols_factory();

        self.records
            .into_par_iter()
            .zip(flat_trace.par_chunks_mut(OVERALL_WIDTH))
            .for_each(|(record, slice)| {
                record_to_row(
                    record,
                    &aux_cols_factory,
                    self.range_checker_chip.as_ref(),
                    slice,
                    &memory,
                );
            });

        let matrix = RowMajorMatrix::new(flat_trace, OVERALL_WIDTH);
        AirProofInput::simple_no_pis(matrix)
    }
}

fn record_to_row<F: PrimeField32>(
    record: JalRangeCheckRecord,
    aux_cols_factory: &MemoryAuxColsFactory<F>,
    range_checker_chip: &VariableRangeCheckerChip,
    slice: &mut [F],
    memory: &OfflineMemory<F>,
) {
    let a_record = memory.record_by_id(record.a_rw);
    let col: &mut JalRangeCheckCols<_> = slice.borrow_mut();
    col.is_jal = F::from_bool(record.is_jal);
    col.is_range_check = F::from_bool(!record.is_jal);
    col.a_pointer = a_record.pointer;
    col.state = ExecutionState {
        pc: F::from_canonical_u32(record.state.pc),
        timestamp: F::from_canonical_u32(record.state.timestamp),
    };
    aux_cols_factory.generate_write_aux(a_record, &mut col.writes_aux);
    col.b = F::from_canonical_u32(record.b);
    if !record.is_jal {
        let a_val = a_record.data_at(0);
        let a_val_u32 = a_val.as_canonical_u32();
        let y = a_val_u32 >> 16;
        let x = a_val_u32 & ((1 << 16) - 1);
        range_checker_chip.add_count(x, record.b as usize);
        range_checker_chip.add_count(y, record.c as usize);
        col.c = F::from_canonical_u32(record.c as u32);
        col.y = F::from_canonical_u32(y);
    }
}
