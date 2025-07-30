use crate::adapters::tracing_write;
use crate::adapters::RV32_CELL_BITS;
use openvm_circuit::{
    arch::*,
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV32_REGISTER_AS,
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32AuipcOpcode::{self, *};
use openvm_stark_backend::p3_air::Air;
use openvm_stark_backend::p3_matrix::Matrix;
use openvm_stark_backend::rap::PartitionedBaseAir;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use std::{
    array::{self, from_fn},
    borrow::{Borrow, BorrowMut},
};
#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Default, Debug)]
pub struct AuipcRecord {
    pub start_baton: (),
    pub inline0_baton1: (),
    pub inline2_prev_data: [u8; 4usize],
    pub inline2_baton1: (),
    pub inline0_start_pc: u32,
    pub inline2_prev_t: u32,
    pub inline0_end_baton: (),
    pub inline0_start_t: u32,
    pub inline0_instruction: TL_Instruction,
}
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TL_Instruction {
    Instruction(u32, [u32; 7usize]),
}
impl Default for TL_Instruction {
    fn default() -> Self {
        Self::Instruction(Default::default(), Default::default())
    }
}
#[derive(Clone, Copy, Debug)]
pub enum TL_Timestamp {
    Timestamp((), u32),
}
impl Default for TL_Timestamp {
    fn default() -> Self {
        Self::Timestamp(Default::default(), Default::default())
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AuipcAir {
    custom_bus_bitwise: u16,
    custom_bus_memory: u16,
    custom_bus_range_check: u16,
    custom_bus_program: u16,
    custom_bus_execution: u16,
}
impl<F: Field> BaseAir<F> for AuipcAir {
    fn width(&self) -> usize {
        19usize
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for AuipcAir {}
impl<F: Field> PartitionedBaseAir<F> for AuipcAir {}
impl<AB: InteractionBuilder> Air<AB> for AuipcAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);
        let constant = |x: isize| {
            let abs = AB::F::from_canonical_usize(x.abs() as usize);
            if x >= 0 {
                abs
            } else {
                -abs
            }
        };
        let constant_expr = |x: isize| AB::Expr::from(constant(x));
        let cell = |i: usize| local[i].into();
        let UF_123_0: AB::F;
        UF_123_0 = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(123);
            result
        };
        let UF_1_0: AB::F;
        UF_1_0 = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(1);
            result
        };
        let UF_13_0: AB::F;
        UF_13_0 = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(13);
            result
        };
        let UF_0_0: AB::F;
        UF_0_0 = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(0);
            result
        };
        let UF_16_0: AB::F;
        UF_16_0 = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(16);
            result
        };
        let U8F_0_0: AB::F;
        U8F_0_0 = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(0 as u32);
            result
        };
        builder.assert_eq(cell(0), cell(0) * cell(0));
        "line 434, column 5: x * x ";
        builder.when(cell(0)).assert_eq(
            (cell(9) + cell(6) - cell(2))
                * constant(256).inverse()
                * (cell(9) + cell(6) - cell(2))
                * constant(256).inverse(),
            (cell(9) + cell(6) - cell(2)) * constant(256).inverse(),
        );
        "line 434, column 5: x * x ";
        builder.when(cell(0)).assert_eq(
            (cell(12)
                - cell(1)
                - (constant_expr(256) * cell(9))
                - (constant_expr(256) * constant(256) * cell(10))
                + cell(8)
                + ((cell(10)
                + cell(7)
                + ((cell(9) + cell(6) - cell(2)) * constant(256).inverse())
                - cell(3))
                * constant(256).inverse())
                - cell(4))
                * constant(256).inverse()
                * (cell(12)
                - cell(1)
                - (constant_expr(256) * cell(9))
                - (constant_expr(256) * constant(256) * cell(10))
                + cell(8)
                + ((cell(10)
                + cell(7)
                + ((cell(9) + cell(6) - cell(2)) * constant(256).inverse())
                - cell(3))
                * constant(256).inverse())
                - cell(4))
                * constant(256).inverse(),
            (cell(12)
                - cell(1)
                - (constant_expr(256) * cell(9))
                - (constant_expr(256) * constant(256) * cell(10))
                + cell(8)
                + ((cell(10)
                + cell(7)
                + ((cell(9) + cell(6) - cell(2)) * constant(256).inverse())
                - cell(3))
                * constant(256).inverse())
                - cell(4))
                * constant(256).inverse(),
        );
        "line 434, column 5: x * x ";
        builder.when(cell(0)).assert_eq(
            (cell(10) + cell(7) + ((cell(9) + cell(6) - cell(2)) * constant(256).inverse())
                - cell(3))
                * constant(256).inverse()
                * (cell(10) + cell(7) + ((cell(9) + cell(6) - cell(2)) * constant(256).inverse())
                - cell(3))
                * constant(256).inverse(),
            (cell(10) + cell(7) + ((cell(9) + cell(6) - cell(2)) * constant(256).inverse())
                - cell(3))
                * constant(256).inverse(),
        );
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [cell(1), cell(2), AB::Expr::from(U8F_0_0), constant_expr(0)],
            constant_expr(1) * cell(0),
            1,
        );
        "line 335, column 5: memory << address_space, pointer, |data), t;";
        builder.push_interaction(
            self.custom_bus_memory,
            [
                AB::Expr::from(UF_1_0),
                cell(5),
                cell(1),
                cell(2),
                cell(3),
                cell(4),
                cell(11),
            ],
            constant_expr(1) * cell(0),
            1,
        );
        "line 334, column 5: memory >> address_space, pointer, prev_data, prev_t;";
        builder.push_interaction(
            self.custom_bus_memory,
            [
                AB::Expr::from(UF_1_0),
                cell(5),
                cell(13),
                cell(14),
                cell(15),
                cell(16),
                cell(17),
            ],
            -(constant_expr(1) * cell(0)),
            1,
        );
        "line 494, column 5: range_check << x, num_bits;";
        builder.push_interaction(
            self.custom_bus_range_check,
            [
                cell(11) - cell(17) - UF_1_0 - (constant_expr(65536) * cell(18)),
                AB::Expr::from(UF_16_0),
            ],
            constant_expr(1) * cell(0),
            1,
        );
        "line 494, column 5: range_check << x, num_bits;";
        builder.push_interaction(
            self.custom_bus_range_check,
            [cell(18), AB::Expr::from(UF_13_0)],
            constant_expr(1) * cell(0),
            1,
        );
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [cell(10), cell(7), AB::Expr::from(U8F_0_0), constant_expr(0)],
            constant_expr(1) * cell(0),
            1,
        );
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [cell(9), cell(6), AB::Expr::from(U8F_0_0), constant_expr(0)],
            constant_expr(1) * cell(0),
            1,
        );
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [
                cell(12)
                    - cell(1)
                    - (constant_expr(256) * cell(9))
                    - (constant_expr(256) * constant(256) * cell(10)),
                cell(8),
                AB::Expr::from(U8F_0_0),
                constant_expr(0),
            ],
            constant_expr(1) * cell(0),
            1,
        );
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [cell(3), cell(4), AB::Expr::from(U8F_0_0), constant_expr(0)],
            constant_expr(1) * cell(0),
            1,
        );
        "line 271, column 5: program << start_pc, instruction;";
        builder.push_interaction(
            self.custom_bus_program,
            [
                cell(12),
                AB::Expr::from(UF_123_0),
                cell(5),
                AB::Expr::from(UF_0_0),
                cell(6)
                    + (constant_expr(256) * cell(7))
                    + (constant_expr(256) * constant(256) * cell(8)),
                AB::Expr::from(UF_1_0),
                AB::Expr::from(UF_0_0),
                AB::Expr::from(UF_0_0),
                AB::Expr::from(UF_0_0),
            ],
            constant_expr(1) * cell(0),
            1,
        );
        "line 269, column 5: execution >> start_pc, start_t;";
        builder.push_interaction(
            self.custom_bus_execution,
            [cell(12), cell(11)],
            -(constant_expr(1) * cell(0)),
            1,
        );
        "line 270, column 5: execution << next_pc, next_t;";
        builder.push_interaction(
            self.custom_bus_execution,
            [cell(12) + UF_1_0, cell(11) + UF_1_0],
            constant_expr(1) * cell(0),
            1,
        );
    }
}
#[derive(Clone, derive_new :: new)]
pub struct AuipcStep {}
impl<F, RA> InstructionExecutor<F, RA> for AuipcStep
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, MultiRowLayout<EmptyMultiRowMetadata>, AuipcRecord>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        todo!()
    }
    fn execute(
        &mut self,
        vm_state: VmStateMut<F, TracingMemory, RA>,
        vm_instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let mut record = vm_state
            .ctx
            .alloc(MultiRowLayout::new(EmptyMultiRowMetadata::new()));
        let argument_0 = ();
        let mut inline0_baton2: () = Default::default();
        let mut imm: u32 = Default::default();
        let mut rd: u32 = Default::default();
        let mut inline0_start_timestamp: TL_Timestamp = Default::default();
        let mut inline0_next_t: u32 = Default::default();
        let mut rd_ptr: u32 = Default::default();
        let mut inline1_y: u32 = Default::default();
        let mut inline1_x: u32 = Default::default();
        let mut inline2_data: u32 = Default::default();
        let mut inline2_baton0: () = Default::default();
        let mut inline0_next_pc: u32 = Default::default();
        let mut pc: u32 = Default::default();
        let mut inline2_next_timestamp: TL_Timestamp = Default::default();
        let mut instruction: TL_Instruction = Default::default();
        let mut inline2_next_t: u32 = Default::default();
        let mut inline2_address_space: u32 = Default::default();
        let mut end_timestamp: TL_Timestamp = Default::default();
        let mut inline0_next_timestamp: TL_Timestamp = Default::default();
        let mut next_pc: u32 = Default::default();
        let mut inline2_inline0_y: u32 = Default::default();
        let mut inline1_z: u32 = Default::default();
        let mut end_baton: () = Default::default();
        let mut inline2_inline0_x: u32 = Default::default();
        let mut inline2_inline0_z: u32 = Default::default();
        let mut start_timestamp: TL_Timestamp = Default::default();
        let mut inline2_timestamp: TL_Timestamp = Default::default();
        let mut inline2_pointer: u32 = Default::default();
        let mut inline2_t: u32 = Default::default();
        let mut inline0_start_baton: () = Default::default();
        let mut real_imm: u32 = Default::default();
        record.start_baton = argument_0;
        "line 3, column 9: start_baton";
        inline0_start_baton = record.start_baton;
        "line 264, column 6: execution_start(start_baton, set {start_pc|, set instruction, set {start_t|, set baton1)" ;
        let mut temp_1: u32 = Default::default();
        let mut temp_2: TL_Instruction = Default::default();
        let mut temp_3: u32 = Default::default();
        temp_1 = *vm_state.pc;
        temp_2 = TL_Instruction::Instruction(
            vm_instruction.opcode.as_usize() as u32,
            [
                vm_instruction.a.as_canonical_u32(),
                vm_instruction.b.as_canonical_u32(),
                vm_instruction.c.as_canonical_u32(),
                vm_instruction.d.as_canonical_u32(),
                vm_instruction.e.as_canonical_u32(),
                vm_instruction.f.as_canonical_u32(),
                vm_instruction.g.as_canonical_u32(),
            ],
        );
        temp_3 = vm_state.memory.timestamp;
        record.inline0_start_pc = temp_1;
        record.inline0_instruction = temp_2;
        record.inline0_start_t = temp_3;
        "line 265, column 5: fix start_timestamp = Timestamp(baton1, start_t);";
        inline0_start_timestamp =
            TL_Timestamp::Timestamp(record.inline0_baton1, record.inline0_start_t);
        "line 4, column 13: start_timestamp";
        start_timestamp = inline0_start_timestamp;
        "line 5, column 13: pc";
        pc = record.inline0_start_pc;
        "line 6, column 13: instruction";
        instruction = record.inline0_instruction;
        "line 14, column 5: Instruction(123UF, [set rd_ptr, 0UF, let imm, 1UF, 0UF, 0UF, 0UF]) = rep instruction;" ;
        if let TL_Instruction::Instruction(temp_4, temp_5) = instruction {
            assert_eq!(temp_4, 123);
            let [temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12] = temp_5;
            rd_ptr = temp_6;
            assert_eq!(temp_7, 0);
            imm = temp_8;
            assert_eq!(temp_9, 1);
            assert_eq!(temp_10, 0);
            assert_eq!(temp_11, 0);
            assert_eq!(temp_12, 0);
        } else {
            panic!();
        }
        "line 17, column 6: lshift({imm|, 8U32, set {real_imm|)";
        let mut temp_13: u32 = Default::default();
        temp_13 = (imm) << (8);
        real_imm = temp_13;
        "line 19, column 6: uadd({pc|, {real_imm|, set {rd|)";
        let mut temp_14: u32 = Default::default();
        temp_14 = (pc) + (real_imm);
        rd = temp_14;
        "line 21, column 48: rd";
        inline2_data = rd;
        "line 21, column 40: rd_ptr";
        inline2_pointer = rd_ptr;
        "line 21, column 35: 1UF";
        inline2_address_space = 1;
        "line 21, column 18: start_timestamp";
        inline2_timestamp = start_timestamp;
        "line 325, column 5: def Timestamp(baton0, t) = timestamp;";
        if let TL_Timestamp::Timestamp(temp_15, temp_16) = inline2_timestamp {
            inline2_baton0 = temp_15;
            inline2_t = temp_16;
        } else {
            panic!();
        }
        "line 329, column 6: e_write_memory(baton0, 1UF, pointer, data, set prev_t, set prev_data, set baton1)" ;
        let mut temp_17: u32 = Default::default();
        let mut temp_18: [u8; 4usize] = Default::default();
        tracing_write(
            vm_state.memory,
            (1),
            (inline2_pointer),
            (inline2_data).to_le_bytes(),
            &mut temp_17,
            &mut temp_18,
        );
        record.inline2_prev_t = temp_17;
        record.inline2_prev_data = temp_18;
        "line 331, column 15: 1UF";
        inline2_inline0_y = 1;
        "line 331, column 12: t";
        inline2_inline0_x = inline2_t;
        "line 410, column 6: uadd({x|, {y|, set {z|)";
        let mut temp_19: u32 = Default::default();
        temp_19 = (inline2_inline0_x) + (inline2_inline0_y);
        inline2_inline0_z = temp_19;
        "line 331, column 24: next_t";
        inline2_next_t = inline2_inline0_z;
        "line 332, column 5: fix next_timestamp = Timestamp(baton1, next_t);";
        inline2_next_timestamp = TL_Timestamp::Timestamp(record.inline2_baton1, inline2_next_t);
        "line 21, column 56: end_timestamp";
        end_timestamp = inline2_next_timestamp;
        "line 11, column 16: 1UF";
        inline1_y = 1;
        "line 11, column 12: pc";
        inline1_x = pc;
        "line 410, column 6: uadd({x|, {y|, set {z|)";
        let mut temp_20: u32 = Default::default();
        temp_20 = (inline1_x) + (inline1_y);
        inline1_z = temp_20;
        "line 11, column 25: next_pc";
        next_pc = inline1_z;
        "line 8, column 9: next_pc";
        inline0_next_pc = next_pc;
        "line 7, column 9: end_timestamp";
        inline0_next_timestamp = end_timestamp;
        "line 266, column 5: def Timestamp(baton2, next_t) = next_timestamp;";
        if let TL_Timestamp::Timestamp(temp_21, temp_22) = inline0_next_timestamp {
            inline0_baton2 = temp_21;
            inline0_next_t = temp_22;
        } else {
            panic!();
        }
        "line 267, column 6: execution_end(baton2, next_pc, set end_baton)";
        *vm_state.pc = (inline0_next_pc);
        "line 9, column 13: end_baton";
        end_baton = record.inline0_end_baton;
        (end_baton);
        Ok(())
    }
}
#[derive(Clone, derive_new :: new)]
pub struct AuipcFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}
impl<F: PrimeField32> TraceFiller<F> for AuipcFiller {
    fn fill_trace_row(&self, _: &MemoryAuxColsFactory<F>, mut row_slice: &mut [F]) {
        let record: &AuipcRecord = unsafe { get_record_from_slice(&mut row_slice, ()) };
        let record = record.clone();
        let mut next_pc: u32 = Default::default();
        let mut inline2_inline1_inline1_y: u32 = Default::default();
        let mut inline2_inline0_z: u32 = Default::default();
        let mut inline0_next_t: u32 = Default::default();
        let mut pc_repped: [u8; 2usize] = Default::default();
        let mut rd: u32 = Default::default();
        let mut inline2_inline1_inline0_x: u32 = Default::default();
        let mut inline2_inline0_y: u32 = Default::default();
        let mut imm_repped: [u8; 3usize] = Default::default();
        let mut inline2_inline1_lower: u32 = Default::default();
        let mut inline9_y: u8 = Default::default();
        let mut inline2_timestamp: TL_Timestamp = Default::default();
        let mut inline2_inline1_upper: u32 = Default::default();
        let mut inline2_inline1_y: u32 = Default::default();
        let mut inline2_inline1_inline2_num_bits: u32 = Default::default();
        let mut inline2_inline1_inline2_x: u32 = Default::default();
        let mut carry3: F = Default::default();
        let mut inline5_x: F = Default::default();
        let mut inline2_next_t: u32 = Default::default();
        let mut inline7_x: u8 = Default::default();
        let mut inline2_next_timestamp: TL_Timestamp = Default::default();
        let mut inline3_x: F = Default::default();
        let mut inline2_inline1_inline0_z: u32 = Default::default();
        let mut instruction: TL_Instruction = Default::default();
        let mut inline0_next_pc: u32 = Default::default();
        let mut inline2_inline1_x: u32 = Default::default();
        let mut inline10_x: u8 = Default::default();
        let mut pc: u32 = Default::default();
        let mut inline10_y: u8 = Default::default();
        let mut inline6_y: u8 = Default::default();
        let mut inline2_pointer: u32 = Default::default();
        let mut inline0_next_timestamp: TL_Timestamp = Default::default();
        let mut end_baton: () = Default::default();
        let mut inline6_x: u8 = Default::default();
        let mut inline2_baton0: () = Default::default();
        let mut inline1_y: u32 = Default::default();
        let mut inline2_inline1_inline1_x: u32 = Default::default();
        let mut imm: u32 = Default::default();
        let mut inline2_t: u32 = Default::default();
        let mut start_timestamp: TL_Timestamp = Default::default();
        let mut inline1_z: u32 = Default::default();
        let mut inline2_inline0_x: u32 = Default::default();
        let mut inline2_inline1_inline3_num_bits: u32 = Default::default();
        let mut carry1: F = Default::default();
        let mut inline4_x: F = Default::default();
        let mut inline2_data: u32 = Default::default();
        let mut inline8_y: u8 = Default::default();
        let mut inline8_x: u8 = Default::default();
        let mut rd_ptr: u32 = Default::default();
        let mut inline0_baton2: () = Default::default();
        let mut inline1_x: u32 = Default::default();
        let mut inline0_start_timestamp: TL_Timestamp = Default::default();
        let mut pc_u32: u32 = Default::default();
        let mut pc_upper_limb: u8 = Default::default();
        let mut carry2: F = Default::default();
        let mut inline2_address_space: u32 = Default::default();
        let mut end_timestamp: TL_Timestamp = Default::default();
        let mut inline7_y: u8 = Default::default();
        let mut inline2_inline1_real_diff: u32 = Default::default();
        let mut inline2_inline1_inline1_z: u32 = Default::default();
        let mut real_imm: u32 = Default::default();
        let mut inline0_start_baton: () = Default::default();
        let mut inline9_x: u8 = Default::default();
        let mut inline2_inline1_diff: u32 = Default::default();
        let mut inline2_inline1_inline0_y: u32 = Default::default();
        let mut inline2_inline1_inline3_x: u32 = Default::default();
        "line 3, column 9: start_baton";
        inline0_start_baton = record.start_baton;
        "line 265, column 5: fix start_timestamp = Timestamp(baton1, start_t);";
        inline0_start_timestamp =
            TL_Timestamp::Timestamp(record.inline0_baton1, record.inline0_start_t);
        "line 4, column 13: start_timestamp";
        start_timestamp = inline0_start_timestamp;
        "line 5, column 13: pc";
        pc = record.inline0_start_pc;
        "line 6, column 13: instruction";
        instruction = record.inline0_instruction;
        "line 14, column 5: Instruction(123UF, [set rd_ptr, 0UF, let imm, 1UF, 0UF, 0UF, 0UF]) = rep instruction;" ;
        if let TL_Instruction::Instruction(temp_1, temp_2) = instruction {
            assert_eq!(temp_1, 123);
            let [temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9] = temp_2;
            rd_ptr = temp_3;
            assert_eq!(temp_4, 0);
            imm = temp_5;
            assert_eq!(temp_6, 1);
            assert_eq!(temp_7, 0);
            assert_eq!(temp_8, 0);
            assert_eq!(temp_9, 0);
        } else {
            panic!();
        }
        "line 17, column 6: lshift({imm|, 8U32, set {real_imm|)";
        let mut temp_10: u32 = Default::default();
        temp_10 = (imm) << (8);
        real_imm = temp_10;
        "line 19, column 6: uadd({pc|, {real_imm|, set {rd|)";
        let mut temp_11: u32 = Default::default();
        temp_11 = (pc) + (real_imm);
        rd = temp_11;
        "line 21, column 48: rd";
        inline2_data = rd;
        "line 21, column 40: rd_ptr";
        inline2_pointer = rd_ptr;
        "line 21, column 35: 1UF";
        inline2_address_space = 1;
        "line 21, column 18: start_timestamp";
        inline2_timestamp = start_timestamp;
        "line 325, column 5: def Timestamp(baton0, t) = timestamp;";
        if let TL_Timestamp::Timestamp(temp_12, temp_13) = inline2_timestamp {
            inline2_baton0 = temp_12;
            inline2_t = temp_13;
        } else {
            panic!();
        }
        "line 331, column 15: 1UF";
        inline2_inline0_y = 1;
        "line 331, column 12: t";
        inline2_inline0_x = inline2_t;
        "line 410, column 6: uadd({x|, {y|, set {z|)";
        let mut temp_14: u32 = Default::default();
        temp_14 = (inline2_inline0_x) + (inline2_inline0_y);
        inline2_inline0_z = temp_14;
        "line 331, column 24: next_t";
        inline2_next_t = inline2_inline0_z;
        "line 332, column 5: fix next_timestamp = Timestamp(baton1, next_t);";
        inline2_next_timestamp = TL_Timestamp::Timestamp(record.inline2_baton1, inline2_next_t);
        "line 21, column 56: end_timestamp";
        end_timestamp = inline2_next_timestamp;
        "line 11, column 16: 1UF";
        inline1_y = 1;
        "line 11, column 12: pc";
        inline1_x = pc;
        "line 410, column 6: uadd({x|, {y|, set {z|)";
        let mut temp_15: u32 = Default::default();
        temp_15 = (inline1_x) + (inline1_y);
        inline1_z = temp_15;
        "line 11, column 25: next_pc";
        next_pc = inline1_z;
        "line 8, column 9: next_pc";
        inline0_next_pc = next_pc;
        "line 7, column 9: end_timestamp";
        inline0_next_timestamp = end_timestamp;
        "line 266, column 5: def Timestamp(baton2, next_t) = next_timestamp;";
        if let TL_Timestamp::Timestamp(temp_16, temp_17) = inline0_next_timestamp {
            inline0_baton2 = temp_16;
            inline0_next_t = temp_17;
        } else {
            panic!();
        }
        "line 9, column 13: end_baton";
        end_baton = record.inline0_end_baton;
        "line 28, column 5: set {pc_u32| = {pc|;";
        pc_u32 = pc;
        "line 34, column 5: def carry1 = ( ||pc_u32)[1]) + ||real_imm)[1]) - ||rd)[1]) ) / 256;";
        carry1 = (({
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = pc_u32.to_le_bytes();
                    result
                }[1usize] as u32,
            );
            result
        } + {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = real_imm.to_le_bytes();
                    result
                }[1usize] as u32,
            );
            result
        }) - ({
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[1usize] as u32,
            );
            result
        })) / (F::from_canonical_u32(256));
        "line 36, column 5: def carry2 = ( ||pc_u32)[2]) + ||real_imm)[2]) + carry1 - ||rd)[2]) ) / 256;" ;
        carry2 = (({
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = pc_u32.to_le_bytes();
                    result
                }[2usize] as u32,
            );
            result
        } + {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = real_imm.to_le_bytes();
                    result
                }[2usize] as u32,
            );
            result
        } + carry1)
            - ({
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[2usize] as u32,
            );
            result
        }))
            / (F::from_canonical_u32(256));
        "line 16, column 5: unalloc<U32B> real_imm;";
        "line 38, column 5: def carry3 = ( ||pc_u32)[3]) + ||real_imm)[3]) + carry2 - ||rd)[3]) ) / 256;" ;
        carry3 = (({
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = pc_u32.to_le_bytes();
                    result
                }[3usize] as u32,
            );
            result
        } + {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = real_imm.to_le_bytes();
                    result
                }[3usize] as u32,
            );
            result
        } + carry2)
            - ({
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[3usize] as u32,
            );
            result
        }))
            / (F::from_canonical_u32(256));
        "line 39, column 13: carry3";
        inline5_x = carry3;
        "line 45, column 38: |real_imm)[3]";
        inline10_y = {
            let result: [u8; 4usize];
            result = real_imm.to_le_bytes();
            result
        }[3usize];
        "line 45, column 25: |pc_u32)[3]";
        inline10_x = {
            let result: [u8; 4usize];
            result = pc_u32.to_le_bytes();
            result
        }[3usize];
        "line 117, column 6: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip
            .request_range((inline10_x) as u32, (inline10_y) as u32);
        "line 25, column 5: [0U8F] ++ set imm_repped = rep |real_imm);";
        let [temp_18, temp_19, temp_20, temp_21] = {
            let result: [u8; 4usize];
            result = real_imm.to_le_bytes();
            result
        };
        let [temp_22] = [temp_18];
        assert_eq!(temp_22, 0);
        imm_repped = [temp_19, temp_20, temp_21];
        "line 44, column 38: |real_imm)[2]";
        inline9_y = {
            let result: [u8; 4usize];
            result = real_imm.to_le_bytes();
            result
        }[2usize];
        "line 44, column 25: |pc_u32)[2]";
        inline9_x = {
            let result: [u8; 4usize];
            result = pc_u32.to_le_bytes();
            result
        }[2usize];
        "line 117, column 6: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip
            .request_range((inline9_x) as u32, (inline9_y) as u32);
        "line 32, column 5: [|rd)[0]] ++ set pc_repped ++ [set pc_upper_limb] = rep |pc_u32);";
        let [temp_23, temp_24, temp_25, temp_26] = {
            let result: [u8; 4usize];
            result = pc_u32.to_le_bytes();
            result
        };
        let [temp_27, temp_28, temp_29] = [temp_23, temp_24, temp_25];
        let [temp_30] = [temp_27];
        assert_eq!(
            temp_30,
            {
                let result: [u8; 4usize];
                result = rd.to_le_bytes();
                result
            }[0usize]
        );
        pc_repped = [temp_28, temp_29];
        let [temp_31] = [temp_26];
        pc_upper_limb = temp_31;
        "line 37, column 13: carry2";
        inline4_x = carry2;
        "line 23, column 5: rep |imm) = ||real_imm)[1]) + (256 * ||real_imm)[2]) ) + (256^2 * ||real_imm)[3]) );" ;
        assert_eq!(
            {
                let result: F;
                result = F::from_canonical_u32(
                    {
                        let result: [u8; 4usize];
                        result = real_imm.to_le_bytes();
                        result
                    }[1usize] as u32,
                );
                result
            } + (F::from_canonical_u32(256))
                * ({
                let result: F;
                result = F::from_canonical_u32(
                    {
                        let result: [u8; 4usize];
                        result = real_imm.to_le_bytes();
                        result
                    }[2usize] as u32,
                );
                result
            })
                + (F::from_canonical_u32(256) * F::from_canonical_u32(256))
                * ({
                let result: F;
                result = F::from_canonical_u32(
                    {
                        let result: [u8; 4usize];
                        result = real_imm.to_le_bytes();
                        result
                    }[3usize] as u32,
                );
                result
            }),
            {
                let result: F;
                result = F::from_canonical_u32(imm);
                result
            }
        );
        "line 43, column 38: |real_imm)[1]";
        inline8_y = {
            let result: [u8; 4usize];
            result = real_imm.to_le_bytes();
            result
        }[1usize];
        "line 43, column 25: |pc_u32)[1]";
        inline8_x = {
            let result: [u8; 4usize];
            result = pc_u32.to_le_bytes();
            result
        }[1usize];
        "line 117, column 6: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip
            .request_range((inline8_x) as u32, (inline8_y) as u32);
        "line 18, column 5: alloc<U32B> rd;";
        "line 13, column 5: alloc<UF> rd_ptr;";
        "line 42, column 34: |rd)[3]";
        inline7_y = {
            let result: [u8; 4usize];
            result = rd.to_le_bytes();
            result
        }[3usize];
        "line 42, column 25: |rd)[2]";
        inline7_x = {
            let result: [u8; 4usize];
            result = rd.to_le_bytes();
            result
        }[2usize];
        "line 117, column 6: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip
            .request_range((inline7_x) as u32, (inline7_y) as u32);
        "line 27, column 5: unalloc<U32B> pc_u32;";
        "line 29, column 5: alloc<[U8F; 2]> pc_repped;";
        "line 41, column 34: |rd)[1]";
        inline6_y = {
            let result: [u8; 4usize];
            result = rd.to_le_bytes();
            result
        }[1usize];
        "line 41, column 25: |rd)[0]";
        inline6_x = {
            let result: [u8; 4usize];
            result = rd.to_le_bytes();
            result
        }[0usize];
        "line 117, column 6: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip
            .request_range((inline6_x) as u32, (inline6_y) as u32);
        "line 35, column 13: carry1";
        inline3_x = carry1;
        "line 31, column 5: rep |pc_upper_limb) = |pc) - ||rd)[0]) - (256 * |pc_repped[0])) - (256^2 * |pc_repped[1]));" ;
        assert_eq!(
            ((({
                let result: F;
                result = F::from_canonical_u32(pc);
                result
            }) - ({
                let result: F;
                result = F::from_canonical_u32(
                    {
                        let result: [u8; 4usize];
                        result = rd.to_le_bytes();
                        result
                    }[0usize] as u32,
                );
                result
            })) - ((F::from_canonical_u32(256))
                * ({
                let result: F;
                result = F::from_canonical_u32(pc_repped[0usize] as u32);
                result
            })))
                - ((F::from_canonical_u32(256) * F::from_canonical_u32(256))
                * ({
                let result: F;
                result = F::from_canonical_u32(pc_repped[1usize] as u32);
                result
            })),
            {
                let result: F;
                result = F::from_canonical_u32(pc_upper_limb as u32);
                result
            }
        );
        "line 30, column 5: unalloc<U8F> pc_upper_limb;";
        "line 24, column 5: alloc<[U8F; 3]> imm_repped;";
        "line 269, column 5: execution >> start_pc, start_t;";
        "line 270, column 5: execution << next_pc, next_t;";
        "line 261, column 5: alloc start_pc;";
        "line 262, column 5: alloc<Baton> baton1;";
        "line 263, column 5: alloc<UF> start_t;";
        "line 271, column 5: program << start_pc, instruction;";
        "line 411, column 5: rep |z) = |x) + |y);";
        assert_eq!(
            {
                let result: F;
                result = F::from_canonical_u32(inline1_x);
                result
            } + {
                let result: F;
                result = F::from_canonical_u32(inline1_y);
                result
            },
            {
                let result: F;
                result = F::from_canonical_u32(inline1_z);
                result
            }
        );
        "line 327, column 5: alloc<Baton> baton1;";
        "line 334, column 5: memory >> address_space, pointer, prev_data, prev_t;";
        "line 326, column 5: alloc<[U8F; 4]> prev_data;";
        "line 337, column 30: t";
        inline2_inline1_y = inline2_t;
        "line 337, column 22: prev_t";
        inline2_inline1_x = record.inline2_prev_t;
        "line 502, column 15: x";
        inline2_inline1_inline0_y = inline2_inline1_x;
        "line 502, column 12: y";
        inline2_inline1_inline0_x = inline2_inline1_y;
        "line 415, column 6: usub({x|, {y|, set {z|)";
        let mut temp_32: u32 = Default::default();
        temp_32 = (inline2_inline1_inline0_x) - (inline2_inline1_inline0_y);
        inline2_inline1_inline0_z = temp_32;
        "line 502, column 22: real_diff";
        inline2_inline1_real_diff = inline2_inline1_inline0_z;
        "line 503, column 23: 1UF";
        inline2_inline1_inline1_y = 1;
        "line 503, column 12: real_diff";
        inline2_inline1_inline1_x = inline2_inline1_real_diff;
        "line 415, column 6: usub({x|, {y|, set {z|)";
        let mut temp_33: u32 = Default::default();
        temp_33 = (inline2_inline1_inline1_x) - (inline2_inline1_inline1_y);
        inline2_inline1_inline1_z = temp_33;
        "line 503, column 32: diff";
        inline2_inline1_diff = inline2_inline1_inline1_z;
        "line 507, column 6: bitwise_and({diff|, 65535U32, set {lower|)";
        let mut temp_34: u32 = Default::default();
        temp_34 = (inline2_inline1_diff) & (65535);
        inline2_inline1_lower = temp_34;
        "line 508, column 6: rshift({diff|, 16U32, set {upper|)";
        let mut temp_35: u32 = Default::default();
        temp_35 = (inline2_inline1_diff) >> (16);
        inline2_inline1_upper = temp_35;
        "line 328, column 5: alloc<UF> prev_t;";
        "line 335, column 5: memory << address_space, pointer, |data), t;";
        "line 411, column 5: rep |z) = |x) + |y);";
        assert_eq!(
            {
                let result: F;
                result = F::from_canonical_u32(inline2_inline0_x);
                result
            } + {
                let result: F;
                result = F::from_canonical_u32(inline2_inline0_y);
                result
            },
            {
                let result: F;
                result = F::from_canonical_u32(inline2_inline0_z);
                result
            }
        );
        "line 509, column 5: rep |lower) = |diff) - (65536 * |upper));";
        assert_eq!(
            ({
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_diff);
                result
            }) - ((F::from_canonical_u32(65536))
                * ({
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_upper);
                result
            })),
            {
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_lower);
                result
            }
        );
        "line 505, column 5: unalloc<UF> lower;";
        "line 506, column 5: alloc<UF> upper;";
        "line 511, column 24: 16UF";
        inline2_inline1_inline2_num_bits = 16;
        "line 511, column 17: lower";
        inline2_inline1_inline2_x = inline2_inline1_lower;
        "line 495, column 6: balance_range_check(x, num_bits)";
        self.range_checker_chip.add_count(
            (inline2_inline1_inline2_x),
            (inline2_inline1_inline2_num_bits) as usize,
        );
        "line 512, column 24: 13UF";
        inline2_inline1_inline3_num_bits = 13;
        "line 512, column 17: upper";
        inline2_inline1_inline3_x = inline2_inline1_upper;
        "line 495, column 6: balance_range_check(x, num_bits)";
        self.range_checker_chip.add_count(
            (inline2_inline1_inline3_x),
            (inline2_inline1_inline3_num_bits) as usize,
        );
        "line 416, column 5: rep |z) = |x) - |y);";
        assert_eq!(
            ({
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_inline0_x);
                result
            }) - ({
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_inline0_y);
                result
            }),
            {
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_inline0_z);
                result
            }
        );
        "line 416, column 5: rep |z) = |x) - |y);";
        assert_eq!(
            ({
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_inline1_x);
                result
            }) - ({
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_inline1_y);
                result
            }),
            {
                let result: F;
                result = F::from_canonical_u32(inline2_inline1_inline1_z);
                result
            }
        );
        "line 494, column 5: range_check << x, num_bits;";
        "line 494, column 5: range_check << x, num_bits;";
        "line 434, column 5: x * x = x;";
        assert_eq!(inline3_x, (inline3_x) * (inline3_x));
        "line 434, column 5: x * x = x;";
        assert_eq!(inline4_x, (inline4_x) * (inline4_x));
        "line 434, column 5: x * x = x;";
        assert_eq!(inline5_x, (inline5_x) * (inline5_x));
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        "line 116, column 5: bitwise << x, y, 0U8F, 0;";
        row_slice[0usize] = F::ONE;
        row_slice[11usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline0_start_t);
            result
        };
        row_slice[17usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline2_prev_t);
            result
        };
        row_slice[5usize] = {
            let result: F;
            result = F::from_canonical_u32(rd_ptr);
            result
        };
        row_slice[6usize] = {
            let result: F;
            result = F::from_canonical_u32(imm_repped[0usize] as u32);
            result
        };
        row_slice[7usize] = {
            let result: F;
            result = F::from_canonical_u32(imm_repped[1usize] as u32);
            result
        };
        row_slice[8usize] = {
            let result: F;
            result = F::from_canonical_u32(imm_repped[2usize] as u32);
            result
        };
        row_slice[18usize] = {
            let result: F;
            result = F::from_canonical_u32(inline2_inline1_upper);
            result
        };
        row_slice[1usize] = {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[0usize] as u32,
            );
            result
        };
        row_slice[2usize] = {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[1usize] as u32,
            );
            result
        };
        row_slice[3usize] = {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[2usize] as u32,
            );
            result
        };
        row_slice[4usize] = {
            let result: F;
            result = F::from_canonical_u32(
                {
                    let result: [u8; 4usize];
                    result = rd.to_le_bytes();
                    result
                }[3usize] as u32,
            );
            result
        };
        row_slice[9usize] = {
            let result: F;
            result = F::from_canonical_u32(pc_repped[0usize] as u32);
            result
        };
        row_slice[10usize] = {
            let result: F;
            result = F::from_canonical_u32(pc_repped[1usize] as u32);
            result
        };
        row_slice[12usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline0_start_pc);
            result
        };
        row_slice[13usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline2_prev_data[0usize] as u32);
            result
        };
        row_slice[14usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline2_prev_data[1usize] as u32);
            result
        };
        row_slice[15usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline2_prev_data[2usize] as u32);
            result
        };
        row_slice[16usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline2_prev_data[3usize] as u32);
            result
        };
    }
}
