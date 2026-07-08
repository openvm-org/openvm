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
use openvm_riscv_transpiler::Rv64JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    ptr_to_u16_limbs, rv64_u32_to_u16_block, Rv64CondRdWriteAdapterExecutor,
    Rv64CondRdWriteAdapterFiller, RV64_PTR_U16_LIMBS, RV_IS_TYPE_IMM_BITS, RV_J_TYPE_IMM_BITS,
    U16_BITS,
};

const LUI_IMM_LOW_BITS: usize = U16_BITS - RV_IS_TYPE_IMM_BITS;
const PC_HIGH_U16_SHIFT: usize = 2 * U16_BITS - PC_BITS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64JalLuiCoreCols<T> {
    pub imm: T,
    // Low 32 bits of rd as u16 cells. Upper register cells are sign extension.
    pub rd_data: [T; RV64_PTR_U16_LIMBS],
    pub imm_low_4: T,
    pub is_jal: T,
    pub is_lui: T,
    pub is_sign_extend: T,
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(Rv64JalLuiCoreCols<u8>)]
pub struct Rv64JalLuiCoreAir {
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

        // LUI: constrain rd = imm << RV_IS_TYPE_IMM_BITS.
        builder
            .when(is_lui)
            .assert_eq(rd[0], imm_low_4 * AB::F::from_u32(1 << RV_IS_TYPE_IMM_BITS));
        builder.when(is_lui).assert_eq(
            imm,
            imm_low_4 + rd[1] * AB::F::from_u32(1 << LUI_IMM_LOW_BITS),
        );
        builder.when(is_jal).assert_zero(imm_low_4);

        // Range-check the low LUI_IMM_LOW_BITS bits of imm.
        self.range_bus
            .range_check(imm_low_4, LUI_IMM_LOW_BITS)
            .eval(builder, is_lui);

        let limb_base = AB::F::from_u32(1 << U16_BITS);

        // JAL: constrain rd_low_32 = from_pc + DEFAULT_PC_STEP.
        builder.when(is_jal).assert_eq(
            rd[0],
            from_pc + AB::F::from_u32(DEFAULT_PC_STEP) - rd[1] * limb_base,
        );

        // Range-check the low 32-bit rd cells.
        self.range_bus
            .range_check(rd[0], U16_BITS)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(rd[1], U16_BITS)
            .eval(builder, is_valid.clone());

        // Tie is_sign_extend to bit 31 of rd.
        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * rd[1] - is_sign_extend * AB::Expr::from_u32(1 << U16_BITS),
                U16_BITS,
            )
            .eval(builder, is_valid.clone());

        // JAL cannot write a return address outside PC_BITS.
        self.range_bus
            .range_check(rd[1] * AB::F::from_u32(1 << PC_HIGH_U16_SHIFT), U16_BITS)
            .eval(builder, is_jal);

        // Sign-extend bit 31 into the upper RV64 register cells.
        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(u16::MAX as u32);
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
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64JalLuiExecutor<A>
where
    F: PrimeField32,
    A: 'static + for<'a> AdapterTraceExecutor<F, ReadData = (), WriteData = [u16; BLOCK_FE_WIDTH]>,
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
        state: VmStateMut<TracingMemory, RA>,
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
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // Rv64JalLuiCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid Rv64JalLuiCoreRecord written by the executor
        // during trace generation
        let record: &Rv64JalLuiCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut Rv64JalLuiCoreCols<F> = core_row.borrow_mut();

        let rd_lo = record.rd_data[0];
        let rd_hi = record.rd_data[1];

        self.range_checker_chip.add_count(rd_lo as u32, U16_BITS);
        self.range_checker_chip.add_count(rd_hi as u32, U16_BITS);
        let is_sign_extend = (rd_hi >> (U16_BITS - 1)) & 1;
        let sign_check = 2u32 * (rd_hi as u32) - (is_sign_extend as u32) * (1 << U16_BITS);
        self.range_checker_chip.add_count(sign_check, U16_BITS);

        let imm_low_4 = if record.is_jal {
            0u8
        } else {
            (record.imm & 0xf) as u8
        };
        if !record.is_jal {
            self.range_checker_chip
                .add_count(imm_low_4 as u32, LUI_IMM_LOW_BITS);
        }

        if record.is_jal {
            self.range_checker_chip
                .add_count((rd_hi as u32) << PC_HIGH_U16_SHIFT, U16_BITS);
        }

        core_row.is_sign_extend = F::from_bool(is_sign_extend != 0);
        core_row.is_lui = F::from_bool(!record.is_jal);
        core_row.is_jal = F::from_bool(record.is_jal);
        core_row.imm_low_4 = F::from_u8(imm_low_4);
        core_row.rd_data = [F::from_u16(rd_lo), F::from_u16(rd_hi)];
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

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jal_lui(is_jal: bool, pc: u32, imm: i32) -> (u32, [u16; BLOCK_FE_WIDTH]) {
    if is_jal {
        let rd_low = pc.wrapping_add(DEFAULT_PC_STEP);
        let next_pc = pc as i32 + imm;
        debug_assert!(next_pc >= 0);
        (next_pc as u32, rv64_u32_to_u16_block(rd_low))
    } else {
        let imm = imm as u32;
        let rd_low = imm << RV_IS_TYPE_IMM_BITS;
        let [lo, hi] = ptr_to_u16_limbs(rd_low);
        let sign = if (hi >> (U16_BITS - 1)) & 1 == 1 {
            u16::MAX
        } else {
            0
        };
        (pc + DEFAULT_PC_STEP, [lo, hi, sign, sign])
    }
}
