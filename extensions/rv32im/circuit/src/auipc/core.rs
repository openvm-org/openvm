use std::{
    array::{self, from_fn},
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
use openvm_rv32im_transpiler::Rv32AuipcOpcode::{self, *};
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

use crate::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32AuipcCoreCols<T> {
    pub is_valid: T,
    // The limbs of the immediate except the least significant limb since it is always 0
    pub imm_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    // The limbs of the PC except the most significant and the least significant limbs
    pub pc_limbs: [T; RV32_REGISTER_NUM_LIMBS - 2],
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS],
}

#[derive(Debug, Clone, Copy, derive_new::new)]
pub struct Rv32AuipcCoreAir {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32AuipcCoreAir {
    fn width(&self) -> usize {
        Rv32AuipcCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32AuipcCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32AuipcCoreAir
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
        let cols: &Rv32AuipcCoreCols<AB::Var> = (*local_core).borrow();

        let Rv32AuipcCoreCols {
            is_valid,
            imm_limbs,
            pc_limbs,
            rd_data,
        } = *cols;
        builder.assert_bool(is_valid);

        // We want to constrain rd = pc + imm (i32 add) where:
        // - rd_data represents limbs of rd
        // - pc_limbs are limbs of pc except the most and least significant limbs
        // - imm_limbs are limbs of imm except the least significant limb

        // We know that rd_data[0] is equal to the least significant limb of PC
        // Thus, the intermediate value will be equal to PC without its most significant limb:
        let intermed_val = rd_data[0]
            + pc_limbs
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                    acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
                });

        // Compute the most significant limb of PC
        let pc_msl = (from_pc - intermed_val)
            * AB::F::from_canonical_usize(1 << (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)))
                .inverse();

        // The vector pc_limbs contains the actual limbs of PC in little endian order
        let pc_limbs = [rd_data[0]]
            .iter()
            .chain(pc_limbs.iter())
            .map(|x| (*x).into())
            .chain([pc_msl])
            .collect::<Vec<AB::Expr>>();

        let mut carry: [AB::Expr; RV32_REGISTER_NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_usize(1 << RV32_CELL_BITS).inverse();

        // Don't need to constrain the least significant limb of the addition
        // since we already know that rd_data[0] = pc_limbs[0] and the least significant limb of imm
        // is 0 Note: imm_limbs doesn't include the least significant limb so imm_limbs[i -
        // 1] means the i-th limb of imm
        for i in 1..RV32_REGISTER_NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (pc_limbs[i].clone() + imm_limbs[i - 1] - rd_data[i] + carry[i - 1].clone());
            builder.when(is_valid).assert_bool(carry[i].clone());
        }

        // Range checking of rd_data entries to RV32_CELL_BITS bits
        for i in 0..(RV32_REGISTER_NUM_LIMBS / 2) {
            self.bus
                .send_range(rd_data[i * 2], rd_data[i * 2 + 1])
                .eval(builder, is_valid);
        }

        // The immediate and PC limbs need range checking to ensure they're within [0,
        // 2^RV32_CELL_BITS) Since we range check two items at a time, doing this way helps
        // efficiently divide the limbs into groups of 2 Note: range checking the limbs of
        // immediate and PC separately would result in additional range checks       since
        // they both have odd number of limbs that need to be range checked
        let mut need_range_check: Vec<AB::Expr> = Vec::new();
        for limb in imm_limbs {
            need_range_check.push(limb.into());
        }

        assert_eq!(pc_limbs.len(), RV32_REGISTER_NUM_LIMBS);
        // use enumerate to match pc_limbs[0] => i = 0, pc_limbs[1] => i = 1, ...
        // pc_limbs[0] is already range checked through rd_data[0], so we skip it
        for (i, limb) in pc_limbs.iter().enumerate().skip(1) {
            // the most significant limb is pc_limbs[3] => i = 3
            if i == pc_limbs.len() - 1 {
                // Range check the most significant limb of pc to be in [0,
                // 2^{PC_BITS-(RV32_REGISTER_NUM_LIMBS-1)*RV32_CELL_BITS})
                need_range_check.push(
                    (*limb).clone()
                        * AB::Expr::from_canonical_usize(
                            1 << (pc_limbs.len() * RV32_CELL_BITS - PC_BITS),
                        ),
                );
            } else {
                need_range_check.push((*limb).clone());
            }
        }

        // need_range_check contains (RV32_REGISTER_NUM_LIMBS - 1) elements from imm_limbs
        // and (RV32_REGISTER_NUM_LIMBS - 1) elements from pc_limbs
        // Hence, is of even length 2*RV32_REGISTER_NUM_LIMBS - 2
        assert_eq!(need_range_check.len() % 2, 0);
        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0].clone(), pair[1].clone())
                .eval(builder, is_valid);
        }

        let imm = imm_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV32_CELL_BITS))
            });
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, AUIPC);
        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [rd_data.map(|x| x.into())].into(),
            instruction: ImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32AuipcOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Rv32AuipcCoreRecord {
    pub from_pc: u32,
    pub imm: u32,
}

pub struct Rv32AuipcStep<F, AdapterAir, AdapterStep> {
    pub air: VmAirWrapper<AdapterAir, Rv32AuipcCoreAir>,
    adapter: AdapterStep,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    mem_helper: SharedMemoryHelper<F>,
}

impl<F, AdapterAir, AdapterStep> Rv32AuipcStep<F, AdapterAir, AdapterStep> {
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        mem_helper: SharedMemoryHelper<F>,
    ) -> Self {
        Self {
            air: VmAirWrapper::new(
                adapter_air,
                Rv32AuipcCoreAir::new(bitwise_lookup_chip.bus()),
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

        let record: &Rv32AuipcCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut Rv32AuipcCoreCols<F> = core_row.borrow_mut();

        let imm_limbs = record.imm.to_le_bytes();
        let pc_limbs = record.from_pc.to_le_bytes();
        let rd_data = run_auipc(record.from_pc, record.imm);
        debug_assert_eq!(imm_limbs[3], 0);

        // range checks:
        // hardcoding for performance: first 3 limbs of imm_limbs, last 3 limbs of pc_limbs where
        // most significant limb of pc_limbs is shifted up
        self.bitwise_lookup_chip
            .request_range(imm_limbs[0] as u32, imm_limbs[1] as u32);
        self.bitwise_lookup_chip
            .request_range(imm_limbs[2] as u32, pc_limbs[1] as u32);
        let msl_shift = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - PC_BITS;
        self.bitwise_lookup_chip
            .request_range(pc_limbs[2] as u32, (pc_limbs[3] as u32) << msl_shift);
        for pair in rd_data.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        // Writing in reverse order
        core_row.rd_data = rd_data.map(F::from_canonical_u8);
        // only the middle 2 limbs:
        core_row.pc_limbs = from_fn(|i| F::from_canonical_u8(pc_limbs[i + 1]));
        core_row.imm_limbs = from_fn(|i| F::from_canonical_u8(imm_limbs[i]));

        core_row.is_valid = F::ONE;
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
    for Rv32AuipcStep<F, AdapterAir, AdapterStep>
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", AUIPC)
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
    for Rv32AuipcStep<F, AdapterAir, AdapterStep>
where
    AdapterStep:
        AdapterTraceStep<F, ReadData = (), WriteData = [u8; RV32_REGISTER_NUM_LIMBS]> + 'static,
    for<'a> RA: RecordArena<
        'a,
        EmptyAdapterCoreLayout<F, AdapterStep>,
        (AdapterStep::RecordMut<'a>, &'a mut Rv32AuipcCoreRecord),
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
        let arena = &mut state.ctx.arenas[chip_index];
        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        AdapterStep::start(*state.pc, state.memory, &mut adapter_record);

        core_record.from_pc = *state.pc;
        core_record.imm = instruction.c.as_canonical_u32();

        let rd = run_auipc(*state.pc, core_record.imm);

        self.adapter
            .write(state.memory, instruction, rd, &mut adapter_record);

        // TODO(ayush): add increment_pc function to vmstate
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, AdapterAir, AdapterStep> InsExecutorE1<F> for Rv32AuipcStep<F, AdapterAir, AdapterStep>
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
        let Instruction { c: imm, .. } = instruction;

        let rd = run_auipc(*state.pc, imm.as_canonical_u32());

        self.adapter.write(state, instruction, rd);

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

impl<F, AdapterAir, AdapterStep> ChipUsageGetter for Rv32AuipcStep<F, AdapterAir, AdapterStep>
where
    F: Field,
    AdapterAir: BaseAir<F>,
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        // TODO(ayush): remove
        1
    }
}
impl<SC, AdapterAir, AdapterStep> Chip<SC> for Rv32AuipcStep<Val<SC>, AdapterAir, AdapterStep>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, Rv32AuipcCoreAir>: Clone + AnyRap<SC> + 'static,
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

// returns rd_data
#[inline(always)]
pub(super) fn run_auipc(pc: u32, imm: u32) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    let rd = pc.wrapping_add(imm << RV32_CELL_BITS);
    rd.to_le_bytes()
}
