use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
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
use openvm_riscv_transpiler::Rv64JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{
    Rv64CondRdWriteAdapterExecutor, Rv64CondRdWriteAdapterFiller, RV64_CELL_BITS,
    RV64_REGISTER_NUM_LIMBS, RV_J_TYPE_IMM_BITS, WORD_NUM_LIMBS,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv64JalLuiCoreCols<T> {
    pub imm: T,
    // We store only the low 32-bit limbs. The high limbs are constrained as sign extension.
    pub rd_data: [T; WORD_NUM_LIMBS],
    pub is_jal: T,
    pub is_lui: T,
    pub is_sign_extend: T,
}

#[derive(Debug, Clone, Copy, derive_new::new)]
pub struct Rv64JalLuiCoreAir {
    pub bus: BitwiseOperationLookupBus,
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
    I::Writes: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
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
            is_jal,
            is_lui,
            is_sign_extend,
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(is_sign_extend);
        builder.when(is_lui).assert_zero(rd[0]);

        for i in 0..WORD_NUM_LIMBS / 2 {
            self.bus
                .send_range(rd[i * 2], rd[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        // - For JAL, enforce rd[3] < 64 by range checking 4 * rd[3].
        // - For LUI, enforce sign-extension selector from the top bit of rd[3]:
        //   2 * rd[3] - 256 * is_sign_extend must be in [0, 255].
        self.bus
            .send_range(
                rd[3] * (AB::Expr::from_canonical_u32(4) * is_jal + is_lui),
                AB::Expr::from_canonical_u32(2) * rd[3]
                    - is_sign_extend * AB::Expr::from_canonical_u32(1 << RV64_CELL_BITS),
            )
            .eval(builder, is_valid.clone());

        let intermed_val = rd
            .iter()
            .skip(1)
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << (i * RV64_CELL_BITS))
            });

        // Constrain that imm * 2^4 is the correct composition of intermed_val in case of LUI
        builder.when(is_lui).assert_eq(
            intermed_val.clone(),
            imm * AB::F::from_canonical_u32(1 << (12 - RV64_CELL_BITS)),
        );

        let intermed_val = rd[0] + intermed_val * AB::Expr::from_canonical_u32(1 << RV64_CELL_BITS);
        // Constrain that from_pc + DEFAULT_PC_STEP is the correct composition of intermed_val in
        // case of JAL
        builder.when(is_jal).assert_eq(
            intermed_val,
            from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP),
        );

        let sign_extend_limb = is_sign_extend * AB::Expr::from_canonical_u32(u8::MAX as u32);
        let rd_data = array::from_fn(|i| {
            if i < WORD_NUM_LIMBS {
                rd[i].into()
            } else {
                sign_extend_limb.clone()
            }
        });

        let to_pc = from_pc + is_lui * AB::F::from_canonical_u32(DEFAULT_PC_STEP) + is_jal * imm;

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_lui * AB::F::from_canonical_u32(LUI as u32)
                + is_jal * AB::F::from_canonical_u32(JAL as u32),
        );

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [].into(),
            writes: [rd_data].into(),
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
    pub rd_data: [u8; RV64_REGISTER_NUM_LIMBS],
    pub is_jal: bool,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalLuiExecutor<A = Rv64CondRdWriteAdapterExecutor> {
    pub adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct Rv64JalLuiFiller<A = Rv64CondRdWriteAdapterFiller> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64JalLuiExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceExecutor<F, ReadData = (), WriteData = [u8; RV64_REGISTER_NUM_LIMBS]>,
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
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // Rv64JalLuiCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid Rv64JalLuiCoreRecord written by the executor
        // during trace generation
        let record: &Rv64JalLuiCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut Rv64JalLuiCoreCols<F> = core_row.borrow_mut();

        for pair in record.rd_data[..WORD_NUM_LIMBS].chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32, pair[1] as u32);
        }
        let is_sign_extend = record.rd_data[3] >> (RV64_CELL_BITS - 1) == 1;
        let second_range_limb =
            (record.rd_data[3] as i32) * 2 - ((is_sign_extend as i32) << RV64_CELL_BITS);
        debug_assert!((0..(1 << RV64_CELL_BITS)).contains(&second_range_limb));
        self.bitwise_lookup_chip.request_range(
            record.rd_data[3] as u32 * (4 * record.is_jal as u32 + (!record.is_jal) as u32),
            second_range_limb as u32,
        );

        // Writing in reverse order
        core_row.is_sign_extend = F::from_bool(is_sign_extend);
        core_row.is_lui = F::from_bool(!record.is_jal);
        core_row.is_jal = F::from_bool(record.is_jal);
        core_row.rd_data = array::from_fn(|i| F::from_canonical_u8(record.rd_data[i]));
        core_row.imm = F::from_canonical_u32(record.imm);
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
pub(super) fn run_jal_lui(is_jal: bool, pc: u32, imm: i32) -> (u32, [u8; RV64_REGISTER_NUM_LIMBS]) {
    if is_jal {
        let mut rd_data = [0u8; RV64_REGISTER_NUM_LIMBS];
        rd_data[..WORD_NUM_LIMBS].copy_from_slice(&(pc + DEFAULT_PC_STEP).to_le_bytes());
        let next_pc = pc as i32 + imm;
        debug_assert!(next_pc >= 0);
        (next_pc as u32, rd_data)
    } else {
        let imm = imm as u32;
        let mut rd_data = [0u8; RV64_REGISTER_NUM_LIMBS];
        rd_data[..WORD_NUM_LIMBS].copy_from_slice(&(imm << 12).to_le_bytes());
        let sign_extend_limb = (rd_data[3] >> (RV64_CELL_BITS - 1)) * u8::MAX;
        rd_data[WORD_NUM_LIMBS..].fill(sign_extend_limb);
        (pc + DEFAULT_PC_STEP, rd_data)
    }
}
