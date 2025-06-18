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
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode::{self, *};
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

use crate::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, RV_J_TYPE_IMM_BITS};

pub(super) const ADDITIONAL_BITS: u32 = 0b11000000;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalLuiCoreCols<T> {
    pub imm: T,
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_jal: T,
    pub is_lui: T,
}

#[derive(Debug, Clone, Copy, derive_new::new)]
pub struct Rv32JalLuiCoreAir {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32JalLuiCoreAir {
    fn width(&self) -> usize {
        Rv32JalLuiCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JalLuiCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32JalLuiCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32JalLuiCoreCols<AB::Var> = (*local_core).borrow();
        let Rv32JalLuiCoreCols::<AB::Var> {
            imm,
            rd_data: rd,
            is_jal,
            is_lui,
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.when(is_lui).assert_zero(rd[0]);

        for i in 0..RV32_REGISTER_NUM_LIMBS / 2 {
            self.bus
                .send_range(rd[i * 2], rd[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        // In case of JAL constrain that last limb has at most [last_limb_bits] bits

        let last_limb_bits = PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        let additional_bits = (last_limb_bits..RV32_CELL_BITS).fold(0, |acc, x| acc + (1 << x));
        let additional_bits = AB::F::from_canonical_u32(additional_bits);
        self.bus
            .send_xor(rd[3], additional_bits, rd[3] + additional_bits)
            .eval(builder, is_jal);

        let intermed_val = rd
            .iter()
            .skip(1)
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });

        // Constrain that imm * 2^4 is the correct composition of intermed_val in case of LUI
        builder.when(is_lui).assert_eq(
            intermed_val.clone(),
            imm * AB::F::from_canonical_u32(1 << (12 - RV32_CELL_BITS)),
        );

        let intermed_val = rd[0] + intermed_val * AB::Expr::from_canonical_u32(1 << RV32_CELL_BITS);
        // Constrain that from_pc + DEFAULT_PC_STEP is the correct composition of intermed_val in
        // case of JAL
        builder.when(is_jal).assert_eq(
            intermed_val,
            from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP),
        );

        let to_pc = from_pc + is_lui * AB::F::from_canonical_u32(DEFAULT_PC_STEP) + is_jal * imm;

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_lui * AB::F::from_canonical_u32(LUI as u32)
                + is_jal * AB::F::from_canonical_u32(JAL as u32),
        );

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [].into(),
            writes: [rd.map(|x| x.into())].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32JalLuiOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32JalLuiStepRecord {
    pub imm: u32,
    pub rd_data: [u8; RV32_REGISTER_NUM_LIMBS],
    pub is_jal: bool,
}

pub struct Rv32JalLuiStep<F, AdapterAir, AdapterStep> {
    pub air: VmAirWrapper<AdapterAir, Rv32JalLuiCoreAir>,
    adapter: AdapterStep,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AdapterAir, AdapterStep> Rv32JalLuiStep<F, AdapterAir, AdapterStep> {
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            air: VmAirWrapper::new(
                adapter_air,
                Rv32JalLuiCoreAir::new(bitwise_lookup_chip.bus()),
            ),
            adapter: adapter_step,
            bitwise_lookup_chip,
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
        let record: &Rv32JalLuiStepRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut Rv32JalLuiCoreCols<F> = core_row.borrow_mut();

        for pair in record.rd_data.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        if record.is_jal {
            self.bitwise_lookup_chip
                .request_xor(record.rd_data[3] as u32, ADDITIONAL_BITS);
        }

        // Writing in reverse order
        core_row.is_lui = F::from_bool(!record.is_jal);
        core_row.is_jal = F::from_bool(record.is_jal);
        core_row.rd_data = record.rd_data.map(F::from_canonical_u8);
        core_row.imm = F::from_canonical_u32(record.imm);
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

impl<F, AdapterAir, AdapterStep> InstructionExecutor<F>
    for Rv32JalLuiStep<F, AdapterAir, AdapterStep>
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JalLuiOpcode::from_usize(opcode - Rv32JalLuiOpcode::CLASS_OFFSET)
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

impl<F, AdapterAir, AdapterStep, RA> InsExecutor<F, RA>
    for Rv32JalLuiStep<F, AdapterAir, AdapterStep>
where
    AdapterStep: 'static
        + for<'a> AdapterTraceStep<F, ReadData = (), WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
    for<'a> RA: RecordArena<
        'a,
        EmptyAdapterCoreLayout<F, AdapterStep>,
        (AdapterStep::RecordMut<'a>, &'a mut Rv32JalLuiStepRecord),
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

        let is_jal = opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET) == JAL as usize;
        let signed_imm = get_signed_imm(is_jal, imm);

        let (to_pc, rd_data) = run_jal_lui(is_jal, *state.pc, signed_imm);

        core_record.imm = imm.as_canonical_u32();
        core_record.rd_data = rd_data;
        core_record.is_jal = is_jal;

        self.adapter
            .write(state.memory, instruction, rd_data, &mut adapter_record);

        *state.pc = to_pc;

        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep> InsExecutorE1<F> for Rv32JalLuiStep<F, AdapterAir, AdapterStep>
where
    F: PrimeField32,
    AdapterStep: 'static
        + for<'a> AdapterExecutorE1<F, ReadData = (), WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
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

        let is_jal = opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET) == JAL as usize;
        let signed_imm = get_signed_imm(is_jal, imm);
        let (to_pc, rd) = run_jal_lui(is_jal, *state.pc, signed_imm);

        self.adapter.write(state, instruction, rd);

        *state.pc = to_pc;

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

impl<F, AdapterAir, AdapterStep> ChipUsageGetter for Rv32JalLuiStep<F, AdapterAir, AdapterStep>
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

impl<SC, AdapterAir, AdapterStep> Chip<SC> for Rv32JalLuiStep<Val<SC>, AdapterAir, AdapterStep>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, Rv32JalLuiCoreAir>: Clone + AnyRap<SC> + 'static,
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

// returns the canonical signed representation of the immediate
// `imm` can be "negative" as a field element
pub(super) fn get_signed_imm<F: PrimeField32>(is_jal: bool, imm: F) -> i32 {
    let imm_f = imm.as_canonical_u32();
    if is_jal {
        if imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)) {
            imm_f as i32
        } else {
            let neg_imm_f = F::ORDER_U32 - imm_f;
            debug_assert!(neg_imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)));
            -(neg_imm_f as i32)
        }
    } else {
        imm_f as i32
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jal_lui(is_jal: bool, pc: u32, imm: i32) -> (u32, [u8; RV32_REGISTER_NUM_LIMBS]) {
    if is_jal {
        let rd_data = (pc + DEFAULT_PC_STEP).to_le_bytes();
        let next_pc = pc as i32 + imm;
        debug_assert!(next_pc >= 0);
        (next_pc as u32, rd_data)
    } else {
        let imm = imm as u32;
        let rd = imm << 12;
        (pc + DEFAULT_PC_STEP, rd.to_le_bytes())
    }
}
