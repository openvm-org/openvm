use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit::{
    arch::{
        get_record_from_slice,
        hasher::{Hasher, HasherChip},
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, EmptyAdapterCoreLayout,
        ExecutionError, ImmInstruction, PreflightExecutor, RecordArena, TraceFiller,
        VmAdapterInterface, VmCoreAir, VmField, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    count::{bus::DeferralCircuitCountBus, DeferralCircuitCountChip},
    poseidon2::{bus::DeferralPoseidon2Bus, DeferralPoseidon2Chip},
    utils::{byte_commit_to_f, f_commit_to_bytes, COMMIT_NUM_BYTES, F_NUM_BYTES},
    DeferralFn,
};

///////////////////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct DeferralCallReads<B, F> {
    // Commit to a specific deferral input, passed in by the user as a pointer
    pub input_commit: [B; COMMIT_NUM_BYTES],

    // Native address space accumulators immediately prior to the current deferral call
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct DeferralCallWrites<B, F> {
    // Output key for raw output + its length in bytes. These bytes are written as one
    // contiguous heap write, with layout [output_commit || output_len_le]. Note output_len
    // **must** be divisible by DIGEST_SIZE.
    pub output_commit: [B; COMMIT_NUM_BYTES],
    pub output_len: [B; F_NUM_BYTES],

    // Native address space accumulators after incorporating the current deferral call
    pub new_input_acc: [F; DIGEST_SIZE],
    pub new_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralCallCoreCols<T> {
    pub is_valid: T,
    pub deferral_idx: T,
    pub reads: DeferralCallReads<T, T>,
    pub writes: DeferralCallWrites<T, T>,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct DeferralCallCoreAir {
    pub count_bus: DeferralCircuitCountBus,
    pub poseidon2_bus: DeferralPoseidon2Bus,
}

impl<F: Field> BaseAir<F> for DeferralCallCoreAir {
    fn width(&self) -> usize {
        DeferralCallCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DeferralCallCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for DeferralCallCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<DeferralCallReads<AB::Expr, AB::Expr>>,
    I::Writes: From<DeferralCallWrites<AB::Expr, AB::Expr>>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &DeferralCallCoreCols<_> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        let input_f_commit = byte_commit_to_f(&cols.reads.input_commit);
        let output_f_commit = byte_commit_to_f(&cols.writes.output_commit);

        self.poseidon2_bus
            .compress(
                cols.reads.old_input_acc,
                input_f_commit,
                cols.writes.new_input_acc,
            )
            .eval(builder, cols.is_valid);

        self.poseidon2_bus
            .compress(
                cols.reads.old_output_acc,
                output_f_commit,
                cols.writes.new_output_acc,
            )
            .eval(builder, cols.is_valid);

        self.count_bus
            .send(cols.deferral_idx)
            .eval(builder, cols.is_valid);

        AdapterAirContext {
            to_pc: None,
            reads: DeferralCallReads {
                input_commit: cols.reads.input_commit.map(Into::into),
                old_input_acc: cols.reads.old_input_acc.map(Into::into),
                old_output_acc: cols.reads.old_output_acc.map(Into::into),
            }
            .into(),
            writes: DeferralCallWrites {
                output_commit: cols.writes.output_commit.map(Into::into),
                output_len: cols.writes.output_len.map(Into::into),
                new_input_acc: cols.writes.new_input_acc.map(Into::into),
                new_output_acc: cols.writes.new_output_acc.map(Into::into),
            }
            .into(),
            instruction: ImmInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_usize(DeferralOpcode::CALL.global_opcode_usize()),
                immediate: cols.deferral_idx.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        DeferralOpcode::CLASS_OFFSET
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// EXECUTION + TRACEGEN
///////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct DeferralCallCoreRecord<F> {
    pub deferral_idx: F,
    pub read_data: DeferralCallReads<u8, F>,
    pub write_data: DeferralCallWrites<u8, F>,
}

#[derive(Clone, derive_new::new)]
pub struct DeferralCallCoreExecutor<A, F: VmField> {
    adapter: A,
    deferral_fns: Vec<Arc<DeferralFn>>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
}

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralCallCoreFiller<A, F: VmField> {
    adapter: A,
    count_chip: Arc<DeferralCircuitCountChip>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
}

impl<F, A, RA> PreflightExecutor<F, RA> for DeferralCallCoreExecutor<A, F>
where
    F: VmField,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = DeferralCallReads<u8, F>,
            WriteData = DeferralCallWrites<u8, F>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, DeferralCallCoreRecord<F>),
    >,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::CALL)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, mut core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        core_record.deferral_idx = instruction.c;

        let read_data = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);
        core_record.read_data = read_data;

        let input_commit = byte_commit_to_f(&read_data.input_commit.map(F::from_u8));

        let def_idx = core_record.deferral_idx.to_unique_u32();
        let (output_commit, output_len) = self.deferral_fns[def_idx as usize].execute(
            &input_commit,
            &mut state.streams.deferrals[def_idx as usize],
            def_idx,
            &self.poseidon2_chip,
        );

        let new_input_acc = self
            .poseidon2_chip
            .compress(&read_data.old_input_acc, &input_commit);
        let new_output_acc = self
            .poseidon2_chip
            .compress(&read_data.old_output_acc, &output_commit);

        let write_data = DeferralCallWrites {
            output_commit: f_commit_to_bytes(&output_commit),
            output_len: output_len.to_le_bytes(),
            new_input_acc,
            new_output_acc,
        };
        core_record.write_data = write_data;
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for DeferralCallCoreFiller<A, F>
where
    F: VmField,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // DeferralCallCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid DeferralCallCoreRecord written by the executor
        // during trace generation
        let record: &DeferralCallCoreRecord<F> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let cols: &mut DeferralCallCoreCols<F> = core_row.borrow_mut();

        self.count_chip
            .add_count(record.deferral_idx.as_canonical_u32());

        let input_f_commit: [F; _] =
            byte_commit_to_f(&record.read_data.input_commit.map(F::from_u8));
        let output_f_commit: [F; _] =
            byte_commit_to_f(&record.write_data.output_commit.map(F::from_u8));
        self.poseidon2_chip
            .compress_and_record(&record.read_data.old_input_acc, &input_f_commit);
        self.poseidon2_chip
            .compress_and_record(&record.read_data.old_output_acc, &output_f_commit);

        // Write columns in reverse order to avoid clobbering the record.
        cols.writes.new_output_acc = record.write_data.new_output_acc;
        cols.writes.new_input_acc = record.write_data.new_input_acc;
        cols.writes.output_len = record.write_data.output_len.map(F::from_u8);
        cols.writes.output_commit = record.write_data.output_commit.map(F::from_u8);
        cols.reads.old_output_acc = record.read_data.old_output_acc;
        cols.reads.old_input_acc = record.read_data.old_input_acc;
        cols.reads.input_commit = record.read_data.input_commit.map(F::from_u8);
        cols.deferral_idx = record.deferral_idx;
        cols.is_valid = F::ONE;
    }
}
