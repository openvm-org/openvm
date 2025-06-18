use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, tracegen::TracegenCtx, E1E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, EmptyAdapterCoreLayout, ExecutionState, InsExecutor, InsExecutorE1,
        InstructionExecutor, RecordArena, Result, Streams, VmAdapterInterface, VmAirWrapper,
        VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryController, SharedMemoryHelper,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    utils::select,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::{ParallelIterator, ParallelSliceMut},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues},
    AirRef, Chip, ChipUsageGetter,
};
use rand::rngs::StdRng;

use crate::adapters::LoadStoreInstruction;

/// LoadSignExtend Core Chip handles byte/halfword into word conversions through sign extend
/// This chip uses read_data to construct write_data
/// prev_data columns are not used in constraints defined in the CoreAir, but are used in
/// constraints by the Adapter shifted_read_data is the read_data shifted by (shift_amount & 2),
/// this reduces the number of opcode flags needed using this shifted data we can generate the
/// write_data as if the shift_amount was 0 for loadh and 0 or 1 for loadb
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadSignExtendCoreCols<T, const NUM_CELLS: usize> {
    /// This chip treats loadb with 0 shift and loadb with 1 shift as different instructions
    pub opcode_loadb_flag0: T,
    pub opcode_loadb_flag1: T,
    pub opcode_loadh_flag: T,

    pub shift_most_sig_bit: T,
    // The bit that is extended to the remaining bits
    pub data_most_sig_bit: T,

    pub shifted_read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone, derive_new::new)]
pub struct LoadSignExtendCoreAir<const NUM_CELLS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field, const NUM_CELLS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>
{
    fn width(&self) -> usize {
        LoadSignExtendCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>
{
}

impl<AB, I, const NUM_CELLS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>
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
        let cols: &LoadSignExtendCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadSignExtendCoreCols::<AB::Var, NUM_CELLS> {
            shifted_read_data,
            prev_data,
            opcode_loadb_flag0: is_loadb0,
            opcode_loadb_flag1: is_loadb1,
            opcode_loadh_flag: is_loadh,
            data_most_sig_bit,
            shift_most_sig_bit,
        } = *cols;

        let flags = [is_loadb0, is_loadb1, is_loadh];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag
        });

        builder.assert_bool(is_valid.clone());
        builder.assert_bool(data_most_sig_bit);
        builder.assert_bool(shift_most_sig_bit);

        let expected_opcode = (is_loadb0 + is_loadb1) * AB::F::from_canonical_u8(LOADB as u8)
            + is_loadh * AB::F::from_canonical_u8(LOADH as u8)
            + AB::Expr::from_canonical_usize(Rv32LoadStoreOpcode::CLASS_OFFSET);

        let limb_mask = data_most_sig_bit * AB::Expr::from_canonical_u32((1 << LIMB_BITS) - 1);

        // there are three parts to write_data:
        // - 1st limb is always shifted_read_data
        // - 2nd to (NUM_CELLS/2)th limbs are read_data if loadh and sign extended if loadb
        // - (NUM_CELLS/2 + 1)th to last limbs are always sign extended limbs
        let write_data: [AB::Expr; NUM_CELLS] = array::from_fn(|i| {
            if i == 0 {
                (is_loadh + is_loadb0) * shifted_read_data[i].into()
                    + is_loadb1 * shifted_read_data[i + 1].into()
            } else if i < NUM_CELLS / 2 {
                shifted_read_data[i] * is_loadh + (is_loadb0 + is_loadb1) * limb_mask.clone()
            } else {
                limb_mask.clone()
            }
        });

        // Constrain that most_sig_bit is correct
        let most_sig_limb = shifted_read_data[0] * is_loadb0
            + shifted_read_data[1] * is_loadb1
            + shifted_read_data[NUM_CELLS / 2 - 1] * is_loadh;

        self.range_bus
            .range_check(
                most_sig_limb
                    - data_most_sig_bit * AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)),
                LIMB_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        // Unshift the shifted_read_data to get the original read_data
        let read_data = array::from_fn(|i| {
            select(
                shift_most_sig_bit,
                shifted_read_data[(i + NUM_CELLS - 2) % NUM_CELLS],
                shifted_read_data[i],
            )
        });
        let load_shift_amount = shift_most_sig_bit * AB::Expr::TWO + is_loadb1;

        AdapterAirContext {
            to_pc: None,
            reads: (prev_data, read_data).into(),
            writes: [write_data].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.clone(),
                opcode: expected_opcode,
                is_load: is_valid,
                load_shift_amount,
                store_shift_amount: AB::Expr::ZERO,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32LoadStoreOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct LoadSignExtendCoreRecord<const NUM_CELLS: usize> {
    pub is_byte: bool,
    pub shift_amount: u8,
    pub read_data: [u8; NUM_CELLS],
    pub prev_data: [u8; NUM_CELLS],
}

pub struct LoadSignExtendStep<
    F,
    AdapterAir,
    AdapterStep,
    const NUM_CELLS: usize,
    const LIMB_BITS: usize,
> {
    pub air: VmAirWrapper<AdapterAir, LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>>,
    pub adapter: AdapterStep,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AdapterAir, AdapterStep, const NUM_CELLS: usize, const LIMB_BITS: usize>
    LoadSignExtendStep<F, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
{
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        range_checker_chip: SharedVariableRangeCheckerChip,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            air: VmAirWrapper::new(
                adapter_air,
                LoadSignExtendCoreAir::new(range_checker_chip.bus()),
            ),
            adapter: adapter_step,
            range_checker_chip,
            mem_helper,
        }
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_CELLS: usize, const LIMB_BITS: usize>
    LoadSignExtendStep<F, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
{
    fn fill_trace_row(&self, row_slice: &mut [F])
    where
        F: PrimeField32,
        AdapterStep: 'static + AdapterTraceFiller<F>,
    {
        let (adapter_row, mut core_row) =
            unsafe { row_slice.split_at_mut_unchecked(AdapterStep::WIDTH) };
        self.adapter
            .fill_trace_row(&self.mem_helper.as_borrowed(), adapter_row);
        let record: &LoadSignExtendCoreRecord<NUM_CELLS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut LoadSignExtendCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        let shift = record.shift_amount;
        let most_sig_limb = if record.is_byte {
            record.read_data[shift as usize]
        } else {
            record.read_data[NUM_CELLS / 2 - 1 + shift as usize]
        };

        let most_sig_bit = most_sig_limb & (1 << 7);
        self.range_checker_chip
            .add_count((most_sig_limb - most_sig_bit) as u32, 7);

        core_row.prev_data = record.prev_data.map(F::from_canonical_u8);
        core_row.shifted_read_data = record.read_data.map(F::from_canonical_u8);
        core_row.shifted_read_data.rotate_left((shift & 2) as usize);

        core_row.data_most_sig_bit = F::from_bool(most_sig_bit != 0);
        core_row.shift_most_sig_bit = F::from_bool(shift & 2 == 2);
        core_row.opcode_loadh_flag = F::from_bool(!record.is_byte);
        core_row.opcode_loadb_flag1 = F::from_bool(record.is_byte && ((shift & 1) == 1));
        core_row.opcode_loadb_flag0 = F::from_bool(record.is_byte && ((shift & 1) == 0));
    }

    fn fill_trace(&self, trace: &mut RowMajorMatrix<F>)
    where
        Self: Send + Sync,
        F: PrimeField32 + Clone + Send + Sync,
        AdapterStep: AdapterTraceFiller<F> + Send + Sync + 'static,
    {
        let rows_used = trace.height();
        let width = trace.width();
        trace.values[..rows_used * width]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_trace_row(row_slice);
            });

        let padded_height = next_power_of_two_or_zero(rows_used);
        trace.pad_to_height(padded_height, F::ZERO);
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_CELLS: usize, const LIMB_BITS: usize>
    InstructionExecutor<F>
    for LoadSignExtendStep<F, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - Rv32LoadStoreOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        _memory: &mut MemoryController<F>,
        _streams: &mut Streams<F>,
        _rng: &mut StdRng,
        _instruction: &Instruction<F>,
        _from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>> {
        unimplemented!()
    }
}

impl<F, AdapterAir, AdapterStep, RA, const NUM_CELLS: usize, const LIMB_BITS: usize>
    InsExecutor<F, RA> for LoadSignExtendStep<F, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
where
    AdapterStep: 'static
        + for<'a> AdapterTraceStep<
            F,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
    for<'a> RA: RecordArena<
        'a,
        EmptyAdapterCoreLayout<F, AdapterStep>,
        (
            AdapterStep::RecordMut<'a>,
            &'a mut LoadSignExtendCoreRecord<NUM_CELLS>,
        ),
    >,
{
    fn execute_tracegen(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, TracegenCtx<RA>>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>
    where
        F: PrimeField32,
    {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let arena = &mut state.ctx.arenas[chip_index];
        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        AdapterStep::start(*state.pc, state.memory, &mut adapter_record);

        let tmp = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.is_byte = local_opcode == LOADB;
        core_record.prev_data = tmp.0 .0.map(|x| x as u8);
        core_record.read_data = tmp.0 .1;
        core_record.shift_amount = tmp.1;

        let write_data = run_write_data_sign_extend(
            local_opcode,
            core_record.read_data,
            core_record.shift_amount as usize,
        );

        self.adapter.write(
            state.memory,
            instruction,
            write_data.map(u32::from),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_CELLS: usize, const LIMB_BITS: usize> InsExecutorE1<F>
    for LoadSignExtendStep<F, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    AdapterStep: 'static
        + AdapterExecutorE1<
            F,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { opcode, .. } = instruction;

        let ((_prev_data, read_data), shift_amount) = self.adapter.read(state, instruction);
        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        let write_data = run_write_data_sign_extend(local_opcode, read_data, shift_amount as usize);

        self.adapter
            .write(state, instruction, write_data.map(u32::from));
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_CELLS: usize, const LIMB_BITS: usize> ChipUsageGetter
    for LoadSignExtendStep<F, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
where
    F: Field,
    AdapterAir: BaseAir<F>,
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn trace_width(&self) -> usize {
        BaseAir::width(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        // TODO(ayush): fix this
        // unimplemented!()
        0
    }
}

impl<SC, AdapterAir, AdapterStep, const NUM_CELLS: usize, const LIMB_BITS: usize> Chip<SC>
    for LoadSignExtendStep<Val<SC>, AdapterAir, AdapterStep, NUM_CELLS, LIMB_BITS>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, LoadSignExtendCoreAir<NUM_CELLS, LIMB_BITS>>:
        Clone + AnyRap<SC> + 'static,
    AdapterStep: AdapterTraceFiller<Val<SC>> + Send + Sync + 'static,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        unimplemented!("generate_air_proof_input isn't implemented")
    }

    fn generate_air_proof_input_with_trace(
        self,
        mut trace: RowMajorMatrix<Val<SC>>,
    ) -> AirProofInput<SC> {
        self.fill_trace(&mut trace);
        assert!(
            trace.height() == 0 || trace.height().is_power_of_two(),
            "Trace height must be a power of two"
        );

        let public_values = vec![];
        AirProofInput::simple(trace, public_values)
    }
}

// Returns write_data
#[inline(always)]
pub(super) fn run_write_data_sign_extend<const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    shift: usize,
) -> [u8; NUM_CELLS] {
    match (opcode, shift) {
        (LOADH, 0) | (LOADH, 2) => {
            let ext = (read_data[NUM_CELLS / 2 - 1 + shift] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i < NUM_CELLS / 2 {
                    read_data[i + shift]
                } else {
                    ext
                }
            })
        }
        (LOADB, 0) | (LOADB, 1) | (LOADB, 2) | (LOADB, 3) => {
            let ext = (read_data[shift] >> 7) * u8::MAX;
            array::from_fn(|i| {
                if i == 0 {
                    read_data[i + shift]
                } else {
                    ext
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
