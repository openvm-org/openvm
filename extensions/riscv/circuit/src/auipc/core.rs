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
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::Rv64AuipcOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    ptr_to_u16_limbs, Rv64RdWriteAdapterExecutor, Rv64RdWriteAdapterFiller, RV64_BYTE_BITS,
    RV64_PTR_U16_LIMBS, U16_BITS,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64AuipcCoreCols<T> {
    pub is_valid: T,
    pub is_sign_extend: T,
    // The immediate is split around the byte shift in AUIPC's `imm << 8`.
    pub imm_low_8: T,
    pub imm_high_16: T,
    pub rd_data: [T; RV64_PTR_U16_LIMBS],
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(Rv64AuipcCoreCols<u8>)]
pub struct Rv64AuipcCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64AuipcCoreAir {
    fn width(&self) -> usize {
        Rv64AuipcCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv64AuipcCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv64AuipcCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv64AuipcCoreCols<AB::Var> = (*local_core).borrow();

        let Rv64AuipcCoreCols {
            is_valid,
            is_sign_extend,
            imm_low_8,
            imm_high_16,
            rd_data,
        } = *cols;
        builder.assert_bool(is_valid);
        builder.assert_bool(is_sign_extend);

        // We want to constrain rd = from_pc + (imm << RV64_BYTE_BITS) where:
        // - rd_data represents the low 32 bits of rd as u16 cells
        // - imm_low_8 is the least significant byte of imm
        // - imm_high_16 is the remaining high 16 bits of imm
        let limb_base = AB::F::from_u32(1 << U16_BITS);
        let carry_divide = limb_base.inverse();
        let imm = imm_low_8 + imm_high_16 * AB::Expr::from_u32(1 << RV64_BYTE_BITS);
        let rd_low_32 = rd_data[0] + rd_data[1] * limb_base;

        // Constrain the low 32-bit addition.
        // `from_pc` is bounded to `PC_BITS` by the program bus.
        let carry_top = (from_pc + imm.clone() * AB::F::from_u32(1 << RV64_BYTE_BITS) - rd_low_32)
            * carry_divide
            * carry_divide;
        builder.when(is_valid).assert_bool(carry_top);

        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * rd_data[1]
                    - is_sign_extend * AB::Expr::from_u32(1 << U16_BITS),
                U16_BITS,
            )
            .eval(builder, is_valid);

        // Range check rd and immediate limbs.
        self.range_bus
            .range_check(rd_data[0], U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[1], U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(imm_low_8, RV64_BYTE_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(imm_high_16, U16_BITS)
            .eval(builder, is_valid);

        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(u16::MAX as u32);
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            rd_data[0].into(),
            rd_data[1].into(),
            sign_extend_cell.clone(),
            sign_extend_cell,
        ];
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, AUIPC);
        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [write_data].into(),
            instruction: ImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv64AuipcOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Rv64AuipcCoreRecord {
    pub from_pc: u32,
    pub imm: u32,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64AuipcExecutor<A = Rv64RdWriteAdapterExecutor> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct Rv64AuipcFiller<A = Rv64RdWriteAdapterFiller> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64AuipcExecutor<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData = (), WriteData = [u16; BLOCK_FE_WIDTH]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut Rv64AuipcCoreRecord),
    >,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{AUIPC:?}")
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        core_record.from_pc = *state.pc;
        core_record.imm = instruction.c.as_canonical_u32();

        let rd = run_auipc(*state.pc, core_record.imm);

        self.adapter
            .write(state.memory, instruction, rd, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for Rv64AuipcFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // Rv64AuipcCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid Rv64AuipcCoreRecord written by the executor
        // during trace generation
        let record: &Rv64AuipcCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut Rv64AuipcCoreCols<F> = core_row.borrow_mut();

        let imm_bytes = record.imm.to_le_bytes();
        debug_assert_eq!(imm_bytes[3], 0);
        let imm_low_8 = imm_bytes[0];
        let imm_high_16 = (imm_bytes[1] as u32) | ((imm_bytes[2] as u32) << RV64_BYTE_BITS);

        let rd_block = run_auipc(record.from_pc, record.imm);
        let rd_u16 = [rd_block[0], rd_block[1]];

        // range checks:
        self.range_checker_chip
            .add_count(imm_low_8 as u32, RV64_BYTE_BITS);
        self.range_checker_chip.add_count(imm_high_16, U16_BITS);
        self.range_checker_chip
            .add_count(rd_u16[0] as u32, U16_BITS);
        self.range_checker_chip
            .add_count(rd_u16[1] as u32, U16_BITS);
        let is_sign_extend = (rd_u16[1] >> (U16_BITS - 1)) & 1;
        let second_range_limb =
            2u32 * (rd_u16[1] as u32) - (is_sign_extend as u32) * (1 << U16_BITS);
        self.range_checker_chip
            .add_count(second_range_limb, U16_BITS);

        // Writing in reverse order
        core_row.rd_data = rd_u16.map(F::from_u16);
        core_row.imm_low_8 = F::from_u8(imm_low_8);
        core_row.imm_high_16 = F::from_u32(imm_high_16);
        core_row.is_sign_extend = F::from_bool(is_sign_extend != 0);
        core_row.is_valid = F::ONE;
    }
}

// returns rd_data
#[inline(always)]
pub(super) fn run_auipc(pc: u32, imm: u32) -> [u16; BLOCK_FE_WIDTH] {
    let rd_low_32 = pc.wrapping_add(imm << RV64_BYTE_BITS);
    let [lo, hi] = ptr_to_u16_limbs(rd_low_32);
    let sign = if (hi >> (U16_BITS - 1)) & 1 == 1 {
        u16::MAX
    } else {
        0
    };
    [lo, hi, sign, sign]
}
