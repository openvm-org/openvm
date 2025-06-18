use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, tracegen::TracegenCtx, E1E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, EmptyAdapterCoreLayout, ExecutionState, ImmInstruction, InsExecutor,
        InsExecutorE1, InstructionExecutor, RecordArena, Result, Streams, VmAdapterInterface,
        VmAirWrapper, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryController, SharedMemoryHelper,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{utils::not, AlignedBytesBorrow};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::{ParallelIterator, ParallelSliceMut},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues},
    AirRef, Chip, ChipUsageGetter,
};
use rand::rngs::StdRng;
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchEqualCoreCols<T, const NUM_LIMBS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    // Boolean result of a op b. Should branch if and only if cmp_result = 1.
    pub cmp_result: T,
    pub imm: T,

    pub opcode_beq_flag: T,
    pub opcode_bne_flag: T,

    pub diff_inv_marker: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct BranchEqualCoreAir<const NUM_LIMBS: usize> {
    offset: usize,
    pc_step: u32,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for BranchEqualCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        BranchEqualCoreCols::<F, NUM_LIMBS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F>
    for BranchEqualCoreAir<NUM_LIMBS>
{
}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for BranchEqualCoreAir<NUM_LIMBS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: Default,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BranchEqualCoreCols<_, NUM_LIMBS> = local.borrow();
        let flags = [cols.opcode_beq_flag, cols.opcode_bne_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let a = &cols.a;
        let b = &cols.b;
        let inv_marker = &cols.diff_inv_marker;

        // 1 if cmp_result indicates a and b are equal, 0 otherwise
        let cmp_eq =
            cols.cmp_result * cols.opcode_beq_flag + not(cols.cmp_result) * cols.opcode_bne_flag;
        let mut sum = cmp_eq.clone();

        // For BEQ, inv_marker is used to check equality of a and b:
        // - If a == b, all inv_marker values must be 0 (sum = 0)
        // - If a != b, inv_marker contains 0s for all positions except ONE position i where a[i] !=
        //   b[i]
        // - At this position, inv_marker[i] contains the multiplicative inverse of (a[i] - b[i])
        // - This ensures inv_marker[i] * (a[i] - b[i]) = 1, making the sum = 1
        // Note: There might be multiple valid inv_marker if a != b.
        // But as long as the trace can provide at least one, that’s sufficient to prove a != b.
        //
        // Note:
        // - If cmp_eq == 0, then it is impossible to have sum != 0 if a == b.
        // - If cmp_eq == 1, then it is impossible for a[i] - b[i] == 0 to pass for all i if a != b.
        for i in 0..NUM_LIMBS {
            sum += (a[i] - b[i]) * inv_marker[i];
            builder.assert_zero(cmp_eq.clone() * (a[i] - b[i]));
        }
        builder.when(is_valid.clone()).assert_one(sum);

        let expected_opcode = flags
            .iter()
            .zip(BranchEqualOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        let to_pc = from_pc
            + cols.cmp_result * cols.imm
            + not(cols.cmp_result) * AB::Expr::from_canonical_u32(self.pc_step);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [cols.a.map(Into::into), cols.b.map(Into::into)].into(),
            writes: Default::default(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: cols.imm.into(),
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
pub struct BranchEqualCoreRecord<const NUM_LIMBS: usize> {
    pub a: [u8; NUM_LIMBS],
    pub b: [u8; NUM_LIMBS],
    pub imm: u32,
    pub local_opcode: u8,
}

pub struct BranchEqualStep<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize> {
    pub air: VmAirWrapper<AdapterAir, BranchEqualCoreAir<NUM_LIMBS>>,
    adapter: AdapterStep,
    pub pc_step: u32,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize>
    BranchEqualStep<F, AdapterAir, AdapterStep, NUM_LIMBS>
{
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        offset: usize,
        pc_step: u32,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            air: VmAirWrapper::new(adapter_air, BranchEqualCoreAir::new(offset, pc_step)),
            adapter: adapter_step,
            pc_step,
            mem_helper,
        }
    }

    fn fill_trace_row(&self, row_slice: &mut [F])
    where
        F: PrimeField32,
        AdapterStep: 'static + AdapterTraceFiller<F>,
    {
        let (adapter_row, mut core_row) =
            unsafe { row_slice.split_at_mut_unchecked(AdapterStep::WIDTH) };
        self.adapter
            .fill_trace_row(&self.mem_helper.as_borrowed(), adapter_row);
        let record: &BranchEqualCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BranchEqualCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();

        let (cmp_result, diff_idx, diff_inv_val) = run_eq::<F, NUM_LIMBS>(
            record.local_opcode == BranchEqualOpcode::BEQ as u8,
            &record.a,
            &record.b,
        );
        core_row.diff_inv_marker = [F::ZERO; NUM_LIMBS];
        core_row.diff_inv_marker[diff_idx] = diff_inv_val;

        core_row.opcode_bne_flag =
            F::from_bool(record.local_opcode == BranchEqualOpcode::BNE as u8);
        core_row.opcode_beq_flag =
            F::from_bool(record.local_opcode == BranchEqualOpcode::BEQ as u8);

        core_row.imm = F::from_canonical_u32(record.imm);
        core_row.cmp_result = F::from_bool(cmp_result);

        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = record.a.map(F::from_canonical_u8);
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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize> InstructionExecutor<F>
    for BranchEqualStep<F, AdapterAir, AdapterStep, NUM_LIMBS>
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            BranchEqualOpcode::from_usize(opcode - self.air.core.offset)
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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, RA> InsExecutor<F, RA>
    for BranchEqualStep<F, AdapterAir, AdapterStep, NUM_LIMBS>
where
    AdapterStep: AdapterTraceStep<F, ReadData = [[u8; NUM_LIMBS]; 2], WriteData = ()> + 'static,
    for<'a> RA: RecordArena<
        'a,
        EmptyAdapterCoreLayout<F, AdapterStep>,
        (
            AdapterStep::RecordMut<'a>,
            &'a mut BranchEqualCoreRecord<NUM_LIMBS>,
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
        let &Instruction { opcode, c: imm, .. } = instruction;

        let branch_eq_opcode =
            BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.air.core.offset));

        let arena = &mut state.ctx.arenas[chip_index];
        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        AdapterStep::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.a = rs1;
        core_record.b = rs2;
        core_record.imm = imm.as_canonical_u32();
        core_record.local_opcode = branch_eq_opcode as u8;

        if fast_run_eq(branch_eq_opcode, &rs1, &rs2) {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize> InsExecutorE1<F>
    for BranchEqualStep<F, AdapterAir, AdapterStep, NUM_LIMBS>
where
    F: PrimeField32,
    AdapterStep:
        'static + for<'a> AdapterExecutorE1<F, ReadData = [[u8; NUM_LIMBS]; 2], WriteData = ()>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let branch_eq_opcode =
            BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.air.core.offset));

        let [rs1, rs2] = self.adapter.read(state, instruction);

        let cmp_result = fast_run_eq(branch_eq_opcode, &rs1, &rs2);

        if cmp_result {
            // TODO(ayush): verify this is fine
            // state.pc = state.pc.wrapping_add(imm.as_canonical_u32());
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize> ChipUsageGetter
    for BranchEqualStep<F, AdapterAir, AdapterStep, NUM_LIMBS>
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

impl<SC, AdapterAir, AdapterStep, const NUM_LIMBS: usize> Chip<SC>
    for BranchEqualStep<Val<SC>, AdapterAir, AdapterStep, NUM_LIMBS>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, BranchEqualCoreAir<NUM_LIMBS>>: Clone + AnyRap<SC> + 'static,
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

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn fast_run_eq<const NUM_LIMBS: usize>(
    local_opcode: BranchEqualOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> bool {
    match local_opcode {
        BranchEqualOpcode::BEQ => x == y,
        BranchEqualOpcode::BNE => x != y,
    }
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn run_eq<F, const NUM_LIMBS: usize>(
    is_beq: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, F)
where
    F: PrimeField32,
{
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                !is_beq,
                i,
                (F::from_canonical_u8(x[i]) - F::from_canonical_u8(y[i])).inverse(),
            );
        }
    }
    (is_beq, 0, F::ZERO)
}
