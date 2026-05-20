use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64JalrOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    expand_to_rv64_block, rv64_bytes_to_u32, rv64_u16_block_to_bytes, Rv64JalrAdapterExecutor,
    Rv64JalrAdapterFiller, RV64_PTR_U16_LIMBS, U16_BITS,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64JalrCoreCols<T> {
    pub imm: T,
    // Low 32 bits of rs1 as u16 limbs.
    pub rs1_data: [T; RV64_PTR_U16_LIMBS],
    // High u16 of the low 32 bits of rd.
    pub rd_data: [T; RV64_PTR_U16_LIMBS - 1],
    pub is_valid: T,

    pub to_pc_least_sig_bit: T,
    /// These are the limbs of `to_pc * 2` after the low-bit split.
    pub to_pc_limbs: [T; 2],
    pub imm_sign: T,
}

#[derive(Debug, Clone, derive_new::new, ColumnsAir)]
#[columns_via(Rv64JalrCoreCols<u8>)]
pub struct Rv64JalrCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64JalrCoreAir {
    fn width(&self) -> usize {
        Rv64JalrCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv64JalrCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv64JalrCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<SignedImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv64JalrCoreCols<AB::Var> = (*local_core).borrow();
        let Rv64JalrCoreCols::<AB::Var> {
            imm,
            rs1_data: rs1,
            rd_data: rd_high,
            is_valid,
            imm_sign,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);

        // To save a column, we only store the high u16 limb of low-32 rd.
        // The low u16 limb is derived from from_pc + DEFAULT_PC_STEP and the
        // stored high limb.
        let rd_low = from_pc + AB::F::from_u32(DEFAULT_PC_STEP)
            - rd_high[0] * AB::F::from_u32(1 << U16_BITS);

        let rd_data_low: [AB::Expr; RV64_PTR_U16_LIMBS] = [rd_low.clone(), rd_high[0].into()];

        // rd_data_low is the low-32-bit decomposition of from_pc + DEFAULT_PC_STEP.
        // The range check on rd_low also ensures that the stored high limb is
        // correct: if it is wrong, rd_low absorbs the error and falls outside
        // [0, 2^U16_BITS).
        // Assumes only from_pc in [0, 2^PC_BITS) is allowed by the program bus.
        self.range_bus
            .range_check(rd_low.clone(), U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_high[0], PC_BITS - U16_BITS)
            .eval(builder, is_valid);

        // Constrain each rs1 u16 cell to be a valid limb.
        // The adapter reads one u16-celled memory block, so the bus sees packed
        // u16 values directly.
        for &v in rs1.iter() {
            self.range_bus
                .range_check(v, U16_BITS)
                .eval(builder, is_valid);
        }

        builder.assert_bool(imm_sign);

        let inv = AB::F::from_u32(1 << U16_BITS).inverse();

        // Constrain to_pc_least_sig_bit + 2 * to_pc_limbs = rs1 + imm as a
        // low-32-bit addition with two u16 limbs. RISC-V explicitly clears the
        // least significant bit of the JALR target.
        builder.assert_bool(to_pc_least_sig_bit);
        let carry = (rs1[0] + imm - to_pc_limbs[0] * AB::F::TWO - to_pc_least_sig_bit) * inv;
        builder.when(is_valid).assert_bool(carry.clone());

        // Sign-extend the 16-bit immediate into the high u16 limb.
        let imm_extend_limb = imm_sign * AB::F::from_u32(u16::MAX as u32);
        let carry = (rs1[1] + imm_extend_limb + carry - to_pc_limbs[1]) * inv;
        builder.when(is_valid).assert_bool(carry);

        // Prevent to_pc overflow. to_pc_limbs[0] is 15 bits because it is
        // multiplied by 2 when reconstructing the aligned target.
        self.range_bus
            .range_check(to_pc_limbs[1], PC_BITS - U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(to_pc_limbs[0], U16_BITS - 1)
            .eval(builder, is_valid);
        let to_pc = to_pc_limbs[0] * AB::F::TWO + to_pc_limbs[1] * AB::F::from_u32(1 << U16_BITS);

        // Zero-extend low-32 rs1/rd at the u16-celled adapter interface.
        let rs1_data = expand_to_rv64_block(&rs1);
        let rd_data = expand_to_rv64_block(&rd_data_low);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JALR);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [rs1_data].into(),
            writes: [rd_data].into(),
            instruction: SignedImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm.into(),
                imm_sign: imm_sign.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv64JalrOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64JalrCoreRecord {
    pub imm: u16,
    pub from_pc: u32,
    pub rs1_val: u32,
    pub imm_sign: bool,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalrExecutor<A = Rv64JalrAdapterExecutor> {
    adapter: A,
}

#[derive(Clone)]
pub struct Rv64JalrFiller<A = Rv64JalrAdapterFiller> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A> Rv64JalrFiller<A> {
    pub fn new(adapter: A, range_checker_chip: SharedVariableRangeCheckerChip) -> Self {
        assert!(range_checker_chip.range_max_bits() >= U16_BITS);
        Self {
            adapter,
            range_checker_chip,
        }
    }
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64JalrExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData = [u16; BLOCK_FE_WIDTH], WriteData = [u16; BLOCK_FE_WIDTH]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut Rv64JalrCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64JalrOpcode::from_usize(opcode - Rv64JalrOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, g, .. } = *instruction;

        debug_assert_eq!(
            opcode.local_opcode_idx(Rv64JalrOpcode::CLASS_OFFSET),
            JALR as usize
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let rs1_data = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);
        let rs1_bytes = rv64_u16_block_to_bytes(rs1_data);
        core_record.rs1_val = rv64_bytes_to_u32(rs1_bytes);

        core_record.imm = c.as_canonical_u32() as u16;
        core_record.imm_sign = g.is_one();
        core_record.from_pc = *state.pc;

        let (to_pc, rd_data) = run_jalr(
            core_record.from_pc,
            core_record.rs1_val,
            core_record.imm,
            core_record.imm_sign,
        );

        self.adapter
            .write(state.memory, instruction, rd_data, &mut adapter_record);

        // RISC-V spec explicitly sets the least significant bit of `to_pc` to 0
        *state.pc = to_pc & !1;

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for Rv64JalrFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // Rv64JalrCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid Rv64JalrCoreRecord written by the executor
        // during trace generation
        let record: &Rv64JalrCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut Rv64JalrCoreCols<F> = core_row.borrow_mut();

        let (to_pc, rd_data) =
            run_jalr(record.from_pc, record.rs1_val, record.imm, record.imm_sign);
        let to_pc_limbs = [(to_pc & (u16::MAX as u32)) >> 1, to_pc >> U16_BITS];
        self.range_checker_chip
            .add_count(to_pc_limbs[0], U16_BITS - 1);
        self.range_checker_chip
            .add_count(to_pc_limbs[1], PC_BITS - U16_BITS);

        let rd_low_u16_lo = rd_data[0];
        let rd_low_u16_hi = rd_data[1];

        self.range_checker_chip
            .add_count(rd_low_u16_lo as u32, U16_BITS);
        self.range_checker_chip
            .add_count(rd_low_u16_hi as u32, PC_BITS - U16_BITS);

        let rs1_low_u16_lo = (record.rs1_val & (u16::MAX as u32)) as u16;
        let rs1_low_u16_hi = (record.rs1_val >> U16_BITS) as u16;
        self.range_checker_chip
            .add_count(rs1_low_u16_lo as u32, U16_BITS);
        self.range_checker_chip
            .add_count(rs1_low_u16_hi as u32, U16_BITS);

        // Write in reverse order
        core_row.imm_sign = F::from_bool(record.imm_sign);
        core_row.to_pc_limbs = to_pc_limbs.map(F::from_u32);
        core_row.to_pc_least_sig_bit = F::from_bool(to_pc & 1 == 1);
        // fill_trace_row is called only on valid rows
        core_row.is_valid = F::ONE;
        core_row.rs1_data = [F::from_u16(rs1_low_u16_lo), F::from_u16(rs1_low_u16_hi)];
        core_row.rd_data = [F::from_u16(rd_low_u16_hi); RV64_PTR_U16_LIMBS - 1];
        core_row.imm = F::from_u16(record.imm);
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jalr(
    pc: u32,
    rs1: u32,
    imm: u16,
    imm_sign: bool,
) -> (u32, [u16; BLOCK_FE_WIDTH]) {
    let to_pc = rs1.wrapping_add(imm as u32 + (imm_sign as u32 * ((u16::MAX as u32) << U16_BITS)));
    assert!(to_pc < (1 << PC_BITS));

    let rd_low_u32 = pc.wrapping_add(DEFAULT_PC_STEP);
    let rd_data = [
        (rd_low_u32 & (u16::MAX as u32)) as u16,
        (rd_low_u32 >> U16_BITS) as u16,
        0,
        0,
    ];
    (to_pc, rd_data)
}
