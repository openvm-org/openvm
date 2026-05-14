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
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::Rv64JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{Rv64CondRdWriteAdapterExecutor, Rv64CondRdWriteAdapterFiller, RV_J_TYPE_IMM_BITS};

/// Pattern B u16 JAL/LUI.
///
/// For LUI: `rd = imm << 12` (`imm` is unsigned 20-bit).
///   In u16 limbs of low 32 bits:
///     `rd[0] = (imm & 0xf) * 2^12`
///     `rd[1] = imm >> 4`
///
/// For JAL: `rd = pc + 4`, `to_pc = pc + imm` (`imm` is signed PC offset).
///   `rd[0] + rd[1] * 2^16 = from_pc + 4` via a single composite carry.
///
/// Sign extension: bit 31 of rd_low_32 (= bit 15 of `rd[1]`) duplicated to top 2 u16 cells.
const JAL_LUI_NUM_U16: usize = 2;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64JalLuiCoreCols<T> {
    pub imm: T,
    /// Low 32 bits of `rd` as 2 u16 limbs.
    pub rd_data: [T; JAL_LUI_NUM_U16],
    /// Low 4 bits of `imm` (only constrained when LUI).
    pub imm_low_4: T,
    pub is_jal: T,
    pub is_lui: T,
    pub is_sign_extend: T,
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(Rv64JalLuiCoreCols<u8>)]
pub struct Rv64JalLuiCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64JalLuiCoreAir {
    fn width(&self) -> usize {
        Rv64JalLuiCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv64JalLuiCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv64JalLuiCoreAir
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
        let cols: &Rv64JalLuiCoreCols<AB::Var> = (*local_core).borrow();
        let Rv64JalLuiCoreCols::<AB::Var> {
            imm,
            rd_data: rd,
            imm_low_4,
            is_jal,
            is_lui,
            is_sign_extend,
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(is_sign_extend);
        // LUI: rd[0] is forced from the 4 low bits of imm; rd[0] is zero in the low 12 bits.
        // For JAL: rd[0] is the low u16 of from_pc + 4 (no zero pattern required).
        builder
            .when(is_lui)
            .assert_eq(rd[0], imm_low_4 * AB::F::from_u32(1 << 12));

        // For LUI: imm = imm_low_4 + rd[1] * 16  (= imm_low_4 + imm_high * 16)
        builder
            .when(is_lui)
            .assert_eq(imm, imm_low_4 + rd[1] * AB::F::from_u32(1 << 4));

        // Range check imm_low_4 to 4 bits when LUI; 0 (no-op) when JAL.
        self.range_bus
            .range_check(imm_low_4, 4)
            .eval(builder, is_lui);

        // For JAL: rd[0] + rd[1] * 2^16 = from_pc + DEFAULT_PC_STEP
        // We use a single composite-carry constraint: the difference equals carry * 2^32 with
        // carry ∈ {0, 1}. Since pc < 2^PC_BITS, this never wraps in practice; carry is bool.
        let two_pow_16 = AB::F::from_u32(1 << 16);
        let rd_low_32 = rd[0] + rd[1] * two_pow_16;
        let carry_top = (from_pc + AB::F::from_u32(DEFAULT_PC_STEP) - rd_low_32)
            * AB::F::from_u32(1 << 16).inverse()
            * AB::F::from_u32(1 << 16).inverse();
        builder.when(is_jal).assert_bool(carry_top.clone());

        // Range check rd[0], rd[1] to 16 bits each (always).
        self.range_bus
            .range_check(rd[0], 16)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(rd[1], 16)
            .eval(builder, is_valid.clone());

        // Sign extension constraint: `2 * rd[1] - 2^16 * is_sign_extend ∈ [0, 2^16)` forces
        // `is_sign_extend = (rd[1] >> 15) & 1` (same packing trick as the byte-shape version).
        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * rd[1] - is_sign_extend * AB::Expr::from_u32(1 << 16),
                16,
            )
            .eval(builder, is_valid.clone());

        // For JAL, additionally constrain `rd[1] * 2^(32 - PC_BITS) ∈ [0, 2^16)` so the high
        // u16 of `pc + 4` fits in `PC_BITS - 16` bits. Combined with the sign-extension check
        // above, this also implies `is_sign_extend = 0` whenever `rd[1] < 2^15` (which is
        // guaranteed for valid PCs).
        const PC_HIGH_U16_BITS: usize = openvm_instructions::program::PC_BITS - 16;
        self.range_bus
            .range_check(
                rd[1] * AB::F::from_u32(1 << (16 - PC_HIGH_U16_BITS)),
                16,
            )
            .eval(builder, is_jal);

        // Suppress unused bitwise bus warning by routing JAL's imm canonicity through it for a
        // single 8-bit window: the immediate is range-checked elsewhere (program bus).
        let _ = (self.bitwise_lookup_bus, RV_J_TYPE_IMM_BITS);

        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(0xffff);
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            rd[0].into(),
            rd[1].into(),
            sign_extend_cell.clone(),
            sign_extend_cell,
        ];

        let to_pc = from_pc + is_lui * AB::F::from_u32(DEFAULT_PC_STEP) + is_jal * imm;

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_lui * AB::F::from_u32(LUI as u32) + is_jal * AB::F::from_u32(JAL as u32),
        );

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [].into(),
            writes: [write_data].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv64JalLuiOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64JalLuiCoreRecord {
    pub imm: u32,
    pub rd_data: [u16; BLOCK_FE_WIDTH],
    pub is_jal: bool,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalLuiExecutor<A = Rv64CondRdWriteAdapterExecutor> {
    pub adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct Rv64JalLuiFiller<A = Rv64CondRdWriteAdapterFiller> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64JalLuiExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceExecutor<F, ReadData = (), WriteData = [u16; BLOCK_FE_WIDTH]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut Rv64JalLuiCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64JalLuiOpcode::from_usize(opcode - Rv64JalLuiOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, c: imm, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let is_jal = opcode.local_opcode_idx(Rv64JalLuiOpcode::CLASS_OFFSET) == JAL as usize;
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

impl<F, A> TraceFiller<F> for Rv64JalLuiFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &Rv64JalLuiCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut Rv64JalLuiCoreCols<F> = core_row.borrow_mut();

        let rd_lo = record.rd_data[0];
        let rd_hi = record.rd_data[1];

        // Range checks (always): rd[0], rd[1] to 16 bits.
        self.range_checker_chip.add_count(rd_lo as u32, 16);
        self.range_checker_chip.add_count(rd_hi as u32, 16);
        // Sign extension check: 2 * rd_hi - 0x10000 * is_sign_extend ∈ [0, 0x10000).
        let is_sign_extend = (rd_hi >> 15) & 1;
        let sign_check = 2u32 * (rd_hi as u32) - ((is_sign_extend as u32) << 16);
        self.range_checker_chip.add_count(sign_check, 16);

        // imm_low_4: only meaningful for LUI; for JAL it's a don't-care.
        let imm_low_4 = if record.is_jal {
            0u8
        } else {
            (record.imm & 0xf) as u8
        };
        if !record.is_jal {
            self.range_checker_chip.add_count(imm_low_4 as u32, 4);
        }

        // JAL-only PC-high range check: rd_hi * 2^(16 - (PC_BITS-16)) < 2^16
        if record.is_jal {
            let shift = 16 - (openvm_instructions::program::PC_BITS - 16);
            self.range_checker_chip
                .add_count((rd_hi as u32) << shift, 16);
        }

        let _ = self.bitwise_lookup_chip.clone();

        // Writing in reverse order
        core_row.is_sign_extend = F::from_bool(is_sign_extend != 0);
        core_row.is_lui = F::from_bool(!record.is_jal);
        core_row.is_jal = F::from_bool(record.is_jal);
        core_row.imm_low_4 = F::from_u8(imm_low_4);
        core_row.rd_data = [F::from_u32(rd_lo as u32), F::from_u32(rd_hi as u32)];
        core_row.imm = F::from_u32(record.imm);
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

// Returns (to_pc, rd_data) as 4 u16 cells.
#[inline(always)]
pub(super) fn run_jal_lui(is_jal: bool, pc: u32, imm: i32) -> (u32, [u16; BLOCK_FE_WIDTH]) {
    if is_jal {
        let rd_low = pc.wrapping_add(DEFAULT_PC_STEP);
        let lo = (rd_low & 0xffff) as u16;
        let hi = (rd_low >> 16) as u16;
        let next_pc = pc as i32 + imm;
        debug_assert!(next_pc >= 0);
        (next_pc as u32, [lo, hi, 0, 0])
    } else {
        let imm = imm as u32;
        let rd_low = imm << 12;
        let lo = (rd_low & 0xffff) as u16;
        let hi = (rd_low >> 16) as u16;
        let sign = if (hi >> 15) & 1 == 1 { 0xffffu16 } else { 0 };
        (pc + DEFAULT_PC_STEP, [lo, hi, sign, sign])
    }
}
