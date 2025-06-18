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
        InstructionExecutor, MinimalInstruction, RecordArena, Result, Streams, VmAdapterInterface,
        VmAirWrapper, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryController, SharedMemoryHelper,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::MulOpcode;
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

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MultiplicationCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct MultiplicationCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: RangeTupleCheckerBus<2>,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        MultiplicationCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &MultiplicationCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // Define carry[i] = (sum_{k=0}^{i} b[k] * c[i - k] + carry[i - 1] - a[i]) / 2^LIMB_BITS.
        // If 0 <= a[i], carry[i] < 2^LIMB_BITS, it can be proven that a[i] = sum_{k=0}^{i} (b[k] *
        // c[i - k]) % 2^LIMB_BITS as necessary.
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_u32(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            let expected_limb = if i == 0 {
                AB::Expr::ZERO
            } else {
                carry[i - 1].clone()
            } + (0..=i).fold(AB::Expr::ZERO, |acc, k| acc + (b[k] * c[i - k]));
            carry[i] = AB::Expr::from(carry_divide) * (expected_limb - a[i]);
        }

        for (a, carry) in a.iter().zip(carry.iter()) {
            self.bus
                .send(vec![(*a).into(), carry.clone()])
                .eval(builder, cols.is_valid);
        }

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, MulOpcode::MUL);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
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
pub struct MultiplicationCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
}

pub struct MultiplicationStep<
    F,
    AdapterAir,
    AdapterStep,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
> {
    pub air: VmAirWrapper<AdapterAir, MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>>,
    pub adapter: AdapterStep,
    pub range_tuple_chip: SharedRangeTupleCheckerChip<2>,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MultiplicationStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        range_tuple_chip: SharedRangeTupleCheckerChip<2>,
        offset: usize,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        // The RangeTupleChecker is used to range check (a[i], carry[i]) pairs where 0 <= i
        // < NUM_LIMBS. a[i] must have LIMB_BITS bits and carry[i] is the sum of i + 1 bytes
        // (with LIMB_BITS bits).
        debug_assert!(
            range_tuple_chip.sizes()[0] == 1 << LIMB_BITS,
            "First element of RangeTupleChecker must have size {}",
            1 << LIMB_BITS
        );
        debug_assert!(
            range_tuple_chip.sizes()[1] >= (1 << LIMB_BITS) * NUM_LIMBS as u32,
            "Second element of RangeTupleChecker must have size of at least {}",
            (1 << LIMB_BITS) * NUM_LIMBS as u32
        );

        Self {
            air: VmAirWrapper::new(
                adapter_air,
                MultiplicationCoreAir::new(*range_tuple_chip.bus(), offset),
            ),
            adapter: adapter_step,
            range_tuple_chip,
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

        let record: &MultiplicationCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut MultiplicationCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let (a, carry) = run_mul::<NUM_LIMBS, LIMB_BITS>(&record.b, &record.c);

        for (a, carry) in a.iter().zip(carry.iter()) {
            self.range_tuple_chip.add_count(&[*a as u32, *carry]);
        }

        // write in reverse order
        core_row.is_valid = F::ONE;
        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = a.map(F::from_canonical_u8);
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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    InstructionExecutor<F>
    for MultiplicationStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", MulOpcode::from_usize(opcode - self.air.core.offset))
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

impl<F, AdapterAir, AdapterStep, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    InsExecutor<F, RA> for MultiplicationStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
where
    AdapterStep: 'static
        + for<'a> AdapterTraceStep<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'a> RA: RecordArena<
        'a,
        EmptyAdapterCoreLayout<F, AdapterStep>,
        (
            AdapterStep::RecordMut<'a>,
            &'a mut MultiplicationCoreRecord<NUM_LIMBS, LIMB_BITS>,
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

        debug_assert_eq!(
            MulOpcode::from_usize(opcode.local_opcode_idx(self.air.core.offset)),
            MulOpcode::MUL
        );
        let arena = &mut state.ctx.arenas[chip_index];
        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        AdapterStep::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (a, _) = run_mul::<NUM_LIMBS, LIMB_BITS>(&rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;

        // TODO(ayush): avoid this conversion
        self.adapter
            .write(state.memory, instruction, [a].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize> InsExecutorE1<F>
    for MultiplicationStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    AdapterStep: 'static
        + for<'a> AdapterExecutorE1<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
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

        // Verify the opcode is MUL
        // TODO(ayush): debug_assert
        assert_eq!(
            MulOpcode::from_usize(opcode.local_opcode_idx(self.air.core.offset)),
            MulOpcode::MUL
        );

        let [rs1, rs2] = self.adapter.read(state, instruction).into();

        let (rd, _) = run_mul::<NUM_LIMBS, LIMB_BITS>(&rs1, &rs2);

        self.adapter.write(state, instruction, [rd].into());

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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize> ChipUsageGetter
    for MultiplicationStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
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
        // TODO(ayush): remove
        1
    }
}

impl<SC, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize> Chip<SC>
    for MultiplicationStep<Val<SC>, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>>:
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

// returns mul, carry
#[inline(always)]
pub(super) fn run_mul<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], [u32; NUM_LIMBS]) {
    let mut result = [0u8; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut res = 0u32;
        if i > 0 {
            res = carry[i - 1];
        }
        for j in 0..=i {
            res += (x[j] as u32) * (y[i - j] as u32);
        }
        carry[i] = res >> LIMB_BITS;
        res %= 1u32 << LIMB_BITS;
        result[i] = res as u8;
    }
    (result, carry)
}
