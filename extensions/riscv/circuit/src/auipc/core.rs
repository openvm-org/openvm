use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64AuipcOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    Rv64RdWriteAdapterExecutor, Rv64RdWriteAdapterFiller, RV64_CELL_BITS, RV64_WORD_NUM_LIMBS,
};

/// Pattern B u16 AUIPC.
///
/// Semantics: `rd = pc + (imm << 8)` where `imm` is 24-bit (3 imm_byte limbs).
///
/// At u16-cell granularity (low 32 bits of `rd` split into 2 u16 limbs `rd[0]`, `rd[1]`):
///
/// - `imm << 8` is a 32-bit value with byte 0 = 0; bytes 1..4 = imm bytes 0..3. As 2 u16 limbs:
///     `sl[0] = imm_b0 * 256` (low byte zero, high byte = imm_b0)
///     `sl[1] = imm_b1 + imm_b2 * 256`
/// - The constraint reduces to:
///     `pc[0] + sl[0] = rd[0] + carry0 * 2^16`
///     `pc[1] + sl[1] + carry0 = rd[1] + carry1 * 2^16`
///     where `pc[0]`, `pc[1]` are the low/high u16 of `pc`, with `pc = pc[0] + pc[1] * 2^16`.
/// - Sign extension on bits 32..64 is uniformly `is_sign_extend * 0xffff` per cell, with
///   `is_sign_extend = (rd[1] >> 15) & 1`.
const AUIPC_NUM_U16: usize = 2;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64AuipcCoreCols<T> {
    pub is_valid: T,
    pub is_sign_extend: T,
    /// 3 byte limbs of `imm` (imm is 24 bits).
    pub imm_limbs: [T; RV64_WORD_NUM_LIMBS - 1],
    /// Low 32 bits of `rd` as 2 u16 limbs.
    pub rd_data: [T; AUIPC_NUM_U16],
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(Rv64AuipcCoreCols<u8>)]
pub struct Rv64AuipcCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
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
            imm_limbs,
            rd_data,
        } = *cols;
        builder.assert_bool(is_valid);
        builder.assert_bool(is_sign_extend);

        // u16 limbs of `imm << 8`.
        let sl_lo = imm_limbs[0] * AB::Expr::from_u32(1 << RV64_CELL_BITS); // imm_b0 * 256
        let sl_hi = imm_limbs[1] + imm_limbs[2] * AB::Expr::from_u32(1 << RV64_CELL_BITS);
        // imm_b1 + imm_b2 * 256

        // Decompose pc into 2 u16 limbs: pc = pc_lo + pc_hi * 2^16.
        // We know from_pc is range-checked by the program bus to fit in PC_BITS; we just need
        // a canonical decomposition. The chip witnesses `rd[0]` and infers carry implicitly via
        // the bool-carry constraint on each row.
        let two_pow_16 = AB::F::from_u32(1 << 16);
        let inv_2_16 = two_pow_16.inverse();

        // carry0 ∈ {0, 1} bool-checked via:
        //   carry0 = (pc_lo + sl_lo - rd[0]) / 2^16
        // We don't have pc_lo as a column. Instead express the sum: the chip will use the
        // composed PC `from_pc = pc_lo + pc_hi * 2^16` via the carry chain plus a witnessed
        // `pc_lo` value. Following the existing AUIPC pattern of deriving pc decomposition
        // from from_pc, we represent the system as:
        //
        //   sum_low = from_pc - pc_hi * 2^16          (= pc_lo)
        //   rd[0] = sum_low + sl_lo - carry0 * 2^16
        //   rd[1] = pc_hi + sl_hi + carry0 - carry1 * 2^16
        //
        // We witness `pc_hi` implicitly: from the final equation `rd[1]` and the carry,
        // pc_hi is recovered. The simpler formulation: derive carry0 from rd[0] and from_pc.
        //
        // Concretely:
        //   carry0 = (from_pc + sl_lo - rd[0] - pc_hi * 2^16) * 2^-16 ∈ {0, 1}
        //   carry1 = (pc_hi + sl_hi + carry0 - rd[1]) * 2^-16 ∈ {0, 1}
        //
        // But pc_hi is a free witness. To eliminate it, observe that
        //   from_pc = pc_lo + pc_hi * 2^16 = (rd[0] - sl_lo + carry0 * 2^16) + pc_hi * 2^16
        // so `pc_hi = (from_pc - rd[0] + sl_lo - carry0 * 2^16) * 2^-16`. Substituting into
        // the second equation gives a single constraint linking rd[0], rd[1], from_pc, and
        // the imm bytes.
        //
        // Simpler: use a single carry-chain on 2 u16 limbs with witness `pc_hi_witness`.
        //
        // We track pc_hi as a witness column to keep constraints degree-1 in the chip. Storing
        // pc_hi adds one column but avoids cubic carries.
        //
        // For minimal column count, we keep no extra pc witness and instead enforce
        //   2^32 * carry_top = from_pc + (imm << 8) - rd_low_32
        // where `rd_low_32 = rd[0] + rd[1] * 2^16` and `carry_top ∈ {0, 1}` (PC + imm fits in
        // 32 bits with at most 1 wrap which is handled by the sign-extension upper limbs).
        let rd_low_32 = rd_data[0] + rd_data[1] * two_pow_16;
        let imm_shifted = sl_lo.clone() + sl_hi.clone() * two_pow_16;
        // carry_top * 2^32 = from_pc + imm_shifted - rd_low_32; carry_top ∈ {0, 1}.
        let carry_top = (from_pc + imm_shifted - rd_low_32) * inv_2_16 * inv_2_16;
        builder.when(is_valid).assert_bool(carry_top.clone());

        // Sign extension constraint: the existing AUIPC AIR forces is_sign_extend to equal the
        // top bit of the most-significant byte of rd_low_32 via:
        //   second_range_limb = 2 * rd[3] - 2^8 * is_sign_extend ∈ [0, 2^8)
        // For u16 limbs, the top bit of rd_low_32 is bit 15 of rd[1]. We enforce:
        //   second_range_limb_u16 = 2 * rd[1] - 2^16 * is_sign_extend ∈ [0, 2^16)
        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * rd_data[1]
                    - is_sign_extend * AB::Expr::from_u32(1 << 16),
                16,
            )
            .eval(builder, is_valid);

        // Range checks: u16 limbs of rd to [0, 2^16) and imm bytes to [0, 256).
        self.range_bus
            .range_check(rd_data[0], 16)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[1], 16)
            .eval(builder, is_valid);
        for &b in imm_limbs.iter() {
            self.bitwise_lookup_bus
                .send_range(b, b)
                .eval(builder, is_valid);
        }

        // imm reconstruction: imm = imm_limbs[0] + imm_limbs[1] * 256 + imm_limbs[2] * 2^16
        let imm = imm_limbs[0]
            + imm_limbs[1] * AB::Expr::from_u32(1 << RV64_CELL_BITS)
            + imm_limbs[2] * AB::Expr::from_u32(1 << (2 * RV64_CELL_BITS));

        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(0xffff);
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            rd_data[0].into(),
            rd_data[1].into(),
            sign_extend_cell.clone(),
            sign_extend_cell,
        ];
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, AUIPC);
        // Suppress unused warning on imports for non-AB paths.
        let _ = PC_BITS;
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
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64AuipcExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData = (), WriteData = [u16; BLOCK_FE_WIDTH]>,
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
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &Rv64AuipcCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut Rv64AuipcCoreCols<F> = core_row.borrow_mut();

        let imm_bytes = record.imm.to_le_bytes();
        debug_assert_eq!(imm_bytes[3], 0);
        let imm_limbs = [imm_bytes[0], imm_bytes[1], imm_bytes[2]];

        let rd_block = run_auipc(record.from_pc, record.imm);
        let rd_u16 = [rd_block[0], rd_block[1]];

        // Range checks:
        for &b in imm_limbs.iter() {
            self.bitwise_lookup_chip.request_range(b as u32, b as u32);
        }
        self.range_checker_chip.add_count(rd_u16[0] as u32, 16);
        self.range_checker_chip.add_count(rd_u16[1] as u32, 16);
        let is_sign_extend = (rd_u16[1] >> 15) & 1;
        let second_range_limb = 2u32 * (rd_u16[1] as u32) - ((is_sign_extend as u32) << 16);
        self.range_checker_chip.add_count(second_range_limb, 16);

        core_row.rd_data = rd_u16.map(|v| F::from_u32(v as u32));
        core_row.imm_limbs = imm_limbs.map(F::from_u8);
        core_row.is_sign_extend = F::from_bool(is_sign_extend != 0);
        core_row.is_valid = F::ONE;
    }
}

// Returns the low 32 bits of `rd` as 2 u16 cells plus 2 sign-extension u16 cells (total 4 cells
// matching `BLOCK_FE_WIDTH`).
#[inline(always)]
pub(super) fn run_auipc(pc: u32, imm: u32) -> [u16; BLOCK_FE_WIDTH] {
    let rd_low_32 = pc.wrapping_add(imm << RV64_CELL_BITS);
    let lo = (rd_low_32 & 0xffff) as u16;
    let hi = (rd_low_32 >> 16) as u16;
    let sign = if (hi >> 15) & 1 == 1 { 0xffffu16 } else { 0 };
    [lo, hi, sign, sign]
}
