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
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_transpiler::BranchLessThanOpcode;
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
pub struct BranchLessThanCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],

    // Boolean result of a op b. Should branch if and only if cmp_result = 1.
    pub cmp_result: T,
    pub imm: T,

    pub opcode_blt_flag: T,
    pub opcode_bltu_flag: T,
    pub opcode_bge_flag: T,
    pub opcode_bgeu_flag: T,

    // Most significant limb of a and b respectively as a field element, will be range
    // checked to be within [-128, 127) if signed and [0, 256) if unsigned.
    pub a_msb_f: T,
    pub b_msb_f: T,

    // 1 if a < b, 0 otherwise.
    pub cmp_lt: T,

    // 1 at the most significant index i such that a[i] != b[i], otherwise 0. If such
    // an i exists, diff_val = b[i] - a[i].
    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct BranchLessThanCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BranchLessThanCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>
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
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BranchLessThanCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_blt_flag,
            cols.opcode_bltu_flag,
            cols.opcode_bge_flag,
            cols.opcode_bgeu_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let lt = cols.opcode_blt_flag + cols.opcode_bltu_flag;
        let ge = cols.opcode_bge_flag + cols.opcode_bgeu_flag;
        let signed = cols.opcode_blt_flag + cols.opcode_bge_flag;
        builder.assert_eq(
            cols.cmp_lt,
            cols.cmp_result * lt.clone() + not(cols.cmp_result) * ge.clone(),
        );

        let a = &cols.a;
        let b = &cols.b;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        // Check if a_msb_f and b_msb_f are signed values of a[NUM_LIMBS - 1] and b[NUM_LIMBS - 1]
        // in prime field F.
        let a_diff = a[NUM_LIMBS - 1] - cols.a_msb_f;
        let b_diff = b[NUM_LIMBS - 1] - cols.b_msb_f;
        builder
            .assert_zero(a_diff.clone() * (AB::Expr::from_canonical_u32(1 << LIMB_BITS) - a_diff));
        builder
            .assert_zero(b_diff.clone() * (AB::Expr::from_canonical_u32(1 << LIMB_BITS) - b_diff));

        for i in (0..NUM_LIMBS).rev() {
            let diff = (if i == NUM_LIMBS - 1 {
                cols.b_msb_f - cols.a_msb_f
            } else {
                b[i] - a[i]
            }) * (AB::Expr::from_canonical_u8(2) * cols.cmp_lt - AB::Expr::ONE);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }
        // - If x != y, then prefix_sum = 1 so marker[i] must be 1 iff i is the first index where
        //   diff != 0. Constrains that diff == diff_val where diff_val is non-zero.
        // - If x == y, then prefix_sum = 0 and cmp_lt = 0. Here, prefix_sum cannot be 1 because all
        //   diff are zero, making diff == diff_val fails.

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(not::<AB::Expr>(prefix_sum.clone()))
            .assert_zero(cols.cmp_lt);

        // Check if a_msb_f and b_msb_f are in [-128, 127) if signed, [0, 256) if unsigned.
        self.bus
            .send_range(
                cols.a_msb_f + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * signed.clone(),
                cols.b_msb_f + AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1)) * signed.clone(),
            )
            .eval(builder, is_valid.clone());

        // Range check to ensure diff_val is non-zero.
        self.bus
            .send_range(cols.diff_val - AB::Expr::ONE, AB::F::ZERO)
            .eval(builder, prefix_sum);

        let expected_opcode = flags
            .iter()
            .zip(BranchLessThanOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        let to_pc = from_pc
            + cols.cmp_result * cols.imm
            + not(cols.cmp_result) * AB::Expr::from_canonical_u32(DEFAULT_PC_STEP);

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
pub struct BranchLessThanCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [u8; NUM_LIMBS],
    pub b: [u8; NUM_LIMBS],
    pub imm: u32,
    pub local_opcode: u8,
}

pub struct BranchLessThanStep<
    F,
    AdapterAir,
    AdapterStep,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
> {
    pub air: VmAirWrapper<AdapterAir, BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>>,
    adapter: AdapterStep,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    BranchLessThanStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    AdapterStep: 'static + AdapterTraceFiller<F>,
{
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            air: VmAirWrapper::new(
                adapter_air,
                BranchLessThanCoreAir::new(bitwise_lookup_chip.bus(), offset),
            ),
            adapter: adapter_step,
            bitwise_lookup_chip,
            mem_helper,
        }
    }

    fn fill_trace_row(&self, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) =
            unsafe { row_slice.split_at_mut_unchecked(AdapterStep::WIDTH) };

        let record: &BranchLessThanCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        self.adapter
            .fill_trace_row(&self.mem_helper.as_borrowed(), adapter_row);
        let core_row: &mut BranchLessThanCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let signed = record.local_opcode == BranchLessThanOpcode::BLT as u8
            || record.local_opcode == BranchLessThanOpcode::BGE as u8;
        let ge_op = record.local_opcode == BranchLessThanOpcode::BGE as u8
            || record.local_opcode == BranchLessThanOpcode::BGEU as u8;

        let (cmp_result, diff_idx, a_sign, b_sign) =
            run_cmp::<NUM_LIMBS, LIMB_BITS>(record.local_opcode, &record.a, &record.b);

        let cmp_lt = cmp_result ^ ge_op;

        // We range check (a_msb_f + 128) and (b_msb_f + 128) if signed,
        // a_msb_f and b_msb_f if not
        let (a_msb_f, a_msb_range) = if a_sign {
            (
                -F::from_canonical_u32((1 << LIMB_BITS) - record.a[NUM_LIMBS - 1] as u32),
                record.a[NUM_LIMBS - 1] as u32 - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u32(record.a[NUM_LIMBS - 1] as u32),
                record.a[NUM_LIMBS - 1] as u32 + ((signed as u32) << (LIMB_BITS - 1)),
            )
        };
        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_canonical_u32((1 << LIMB_BITS) - record.b[NUM_LIMBS - 1] as u32),
                record.b[NUM_LIMBS - 1] as u32 - (1 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_canonical_u32(record.b[NUM_LIMBS - 1] as u32),
                record.b[NUM_LIMBS - 1] as u32 + ((signed as u32) << (LIMB_BITS - 1)),
            )
        };

        core_row.diff_val = if diff_idx == NUM_LIMBS {
            F::ZERO
        } else if diff_idx == (NUM_LIMBS - 1) {
            if cmp_lt {
                b_msb_f - a_msb_f
            } else {
                a_msb_f - b_msb_f
            }
        } else if cmp_lt {
            F::from_canonical_u8(record.b[diff_idx] - record.a[diff_idx])
        } else {
            F::from_canonical_u8(record.a[diff_idx] - record.b[diff_idx])
        };

        self.bitwise_lookup_chip
            .request_range(a_msb_range, b_msb_range);

        core_row.diff_marker = [F::ZERO; NUM_LIMBS];

        if diff_idx != NUM_LIMBS {
            self.bitwise_lookup_chip
                .request_range(core_row.diff_val.as_canonical_u32() - 1, 0);
            core_row.diff_marker[diff_idx] = F::ONE;
        }

        core_row.cmp_lt = F::from_bool(cmp_lt);
        core_row.b_msb_f = b_msb_f;
        core_row.a_msb_f = a_msb_f;
        core_row.opcode_bgeu_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BGEU as u8);
        core_row.opcode_bge_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BGE as u8);
        core_row.opcode_bltu_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BLTU as u8);
        core_row.opcode_blt_flag =
            F::from_bool(record.local_opcode == BranchLessThanOpcode::BLT as u8);

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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    InstructionExecutor<F>
    for BranchLessThanStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            BranchLessThanOpcode::from_usize(opcode - self.air.core.offset)
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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize, RA>
    InsExecutor<F, RA> for BranchLessThanStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
where
    AdapterStep: AdapterTraceStep<F, ReadData = [[u8; NUM_LIMBS]; 2], WriteData = ()> + 'static,
    for<'a> RA: RecordArena<
        'a,
        EmptyAdapterCoreLayout<F, AdapterStep>,
        (
            AdapterStep::RecordMut<'a>,
            &'a mut BranchLessThanCoreRecord<NUM_LIMBS, LIMB_BITS>,
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

        let arena = &mut state.ctx.arenas[chip_index];
        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        AdapterStep::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.a = rs1;
        core_record.b = rs2;
        core_record.imm = imm.as_canonical_u32();
        core_record.local_opcode = opcode.local_opcode_idx(self.air.core.offset) as u8;

        if run_cmp::<NUM_LIMBS, LIMB_BITS>(core_record.local_opcode, &rs1, &rs2).0 {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize> InsExecutorE1<F>
    for BranchLessThanStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
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
        let [rs1, rs2] = self.adapter.read(state, instruction);

        // TODO(ayush): probably don't need the other values
        let (cmp_result, _, _, _) = run_cmp::<NUM_LIMBS, LIMB_BITS>(
            opcode.local_opcode_idx(self.air.core.offset) as u8,
            &rs1,
            &rs2,
        );

        if cmp_result {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
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

impl<F, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize> ChipUsageGetter
    for BranchLessThanStep<F, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
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

impl<SC, AdapterAir, AdapterStep, const NUM_LIMBS: usize, const LIMB_BITS: usize> Chip<SC>
    for BranchLessThanStep<Val<SC>, AdapterAir, AdapterStep, NUM_LIMBS, LIMB_BITS>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, BranchLessThanCoreAir<NUM_LIMBS, LIMB_BITS>>:
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

// Returns (cmp_result, diff_idx, x_sign, y_sign)
#[inline(always)]
pub(super) fn run_cmp<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    local_opcode: u8,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, bool, bool) {
    let signed = local_opcode == BranchLessThanOpcode::BLT as u8
        || local_opcode == BranchLessThanOpcode::BGE as u8;
    let ge_op = local_opcode == BranchLessThanOpcode::BGE as u8
        || local_opcode == BranchLessThanOpcode::BGEU as u8;
    let x_sign = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    let y_sign = (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1) && signed;
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            return ((x[i] < y[i]) ^ x_sign ^ y_sign ^ ge_op, i, x_sign, y_sign);
        }
    }
    (ge_op, NUM_LIMBS, x_sign, y_sign)
}
