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
pub type Rv32AuipcDazlChip<F> = VmChipWrapper<F, Rv32AuipcDazlFiller>;
#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Default, Debug)]
pub struct Rv32AuipcDazlRecord {
    name_instruction: TL_Instruction,
    inline_2_1_name_real_diff: u32,
    inline_2_name_prev_data: [u8; 4usize],
    name_pc_bytes: [u8; 4usize],
    inline_2_name_next_t: u32,
    inline_2_name_prev_t: u32,
    inline_0_name_start_t: u32,
    inline_0_name_baton1: (),
    inline_2_1_name_diff: u32,
    name_imm: u32,
    inline_2_name_baton1: (),
    inline_2_1_name_lower: u32,
    name_pc: u32,
    name_next_pc: u32,
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
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TL_Instruction {
    Instruction(u32, [u32; 7usize]),
}
impl Default for TL_Instruction {
    fn default() -> Self {
        Self::Instruction(Default::default(), Default::default())
    }
}
#[derive(derive_new :: new, Clone, Copy, Debug, Default)]
pub struct Rv32AuipcDazlAir {
    pub custom_bus_range_check: u16,
    pub custom_bus_memory: u16,
    pub custom_bus_bitwise: u16,
    pub custom_bus_exe: u16,
    pub custom_bus_program: u16,
}
impl<F: Field> BaseAir<F> for Rv32AuipcDazlAir {
    fn width(&self) -> usize {
        19usize
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for Rv32AuipcDazlAir {}
impl<F: Field> PartitionedBaseAir<F> for Rv32AuipcDazlAir {}
impl<AB: InteractionBuilder> Air<AB> for Rv32AuipcDazlAir {
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
        let scope = cell(0);
        let inline_0_name_start_t = cell(1);
        let anonymous_100_line_0_column_0 = inline_0_name_start_t.clone();
        let anonymous_140_line_0_column_0 = constant_expr(0);
        let name_rd_bytes = [cell(2), cell(3), cell(4), cell(5)];
        let anonymous_86_line_0_column_0 = name_rd_bytes[0usize].clone();
        let name_imm_bytes = [cell(6), cell(7), cell(8)];
        let anonymous_64_line_0_column_0 = name_rd_bytes[0usize].clone();
        let anonymous_40_line_0_column_0 = name_imm_bytes[0usize].clone();
        let anonymous_63_line_0_column_0 = anonymous_64_line_0_column_0.clone();
        let name_pc_middle = [cell(9), cell(10)];
        let anonymous_62_line_0_column_0 = [
            anonymous_63_line_0_column_0.clone(),
            name_pc_middle[0usize].clone(),
            name_pc_middle[1usize].clone(),
        ];
        let name_pc = cell(11);
        let anonymous_53_line_0_column_0 = name_rd_bytes[0usize].clone();
        let anonymous_52_line_0_column_0 = name_pc.clone() - anonymous_53_line_0_column_0.clone();
        let anonymous_55_line_0_column_0 = constant_expr(256);
        let anonymous_56_line_0_column_0 = name_pc_middle[0usize].clone();
        let anonymous_54_line_0_column_0 =
            anonymous_55_line_0_column_0.clone() * anonymous_56_line_0_column_0.clone();
        let anonymous_51_line_0_column_0 =
            anonymous_52_line_0_column_0.clone() - anonymous_54_line_0_column_0.clone();
        let anonymous_59_line_0_column_0 = constant_expr(256);
        let anonymous_58_line_0_column_0 =
            anonymous_59_line_0_column_0.clone() * anonymous_59_line_0_column_0.clone();
        let anonymous_60_line_0_column_0 = name_pc_middle[1usize].clone();
        let anonymous_57_line_0_column_0 =
            anonymous_58_line_0_column_0.clone() * anonymous_60_line_0_column_0.clone();
        let anonymous_50_line_0_column_0 =
            anonymous_51_line_0_column_0.clone() - anonymous_57_line_0_column_0.clone();
        let anonymous_49_line_0_column_0 =
            anonymous_50_line_0_column_0.clone() * constant(16777216).inverse();
        let name_pc_upper_limb = anonymous_49_line_0_column_0.clone();
        let anonymous_65_line_0_column_0 = name_pc_upper_limb.clone();
        let anonymous_61_line_0_column_0 = [
            anonymous_62_line_0_column_0[0usize].clone(),
            anonymous_62_line_0_column_0[1usize].clone(),
            anonymous_62_line_0_column_0[2usize].clone(),
            anonymous_65_line_0_column_0.clone(),
        ];
        let name_pc_bytes = [
            anonymous_61_line_0_column_0[0usize].clone(),
            anonymous_61_line_0_column_0[1usize].clone(),
            anonymous_61_line_0_column_0[2usize].clone(),
            anonymous_61_line_0_column_0[3usize].clone(),
        ];
        let anonymous_92_line_0_column_0 = name_pc_bytes[2usize].clone();
        let anonymous_69_line_0_column_0 = name_pc_bytes[1usize].clone();
        let name_start_timestamp = anonymous_100_line_0_column_0.clone();
        let anonymous_109_line_0_column_0 = name_start_timestamp.clone();
        let inline_2_name_t = anonymous_109_line_0_column_0.clone();
        let temp: AB::F;
        temp = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(1);
            result
        };
        let constant_UF_1 = AB::Expr::from(temp.into());
        let anonymous_111_line_0_column_0 = constant_UF_1.clone();
        let anonymous_113_line_0_column_0 =
            inline_2_name_t.clone() + anonymous_111_line_0_column_0.clone();
        let inline_2_name_next_t = anonymous_113_line_0_column_0.clone();
        let anonymous_112_line_0_column_0 = inline_2_name_next_t.clone();
        let name_end_timestamp = anonymous_112_line_0_column_0.clone();
        let anonymous_101_line_0_column_0 = name_end_timestamp.clone();
        let anonymous_29_line_0_column_0 = constant_expr(0);
        let anonymous_70_line_0_column_0 = name_imm_bytes[0usize].clone();
        let anonymous_68_line_0_column_0 =
            anonymous_69_line_0_column_0.clone() + anonymous_70_line_0_column_0.clone();
        let anonymous_71_line_0_column_0 = name_rd_bytes[1usize].clone();
        let anonymous_67_line_0_column_0 =
            anonymous_68_line_0_column_0.clone() - anonymous_71_line_0_column_0.clone();
        let anonymous_66_line_0_column_0 =
            anonymous_67_line_0_column_0.clone() * constant(256).inverse();
        let anonymous_42_line_0_column_0 = constant_expr(256);
        let anonymous_43_line_0_column_0 = name_imm_bytes[1usize].clone();
        let anonymous_41_line_0_column_0 =
            anonymous_42_line_0_column_0.clone() * anonymous_43_line_0_column_0.clone();
        let anonymous_39_line_0_column_0 =
            anonymous_40_line_0_column_0.clone() + anonymous_41_line_0_column_0.clone();
        let name_rd_ptr = cell(12);
        let anonymous_46_line_0_column_0 = constant_expr(256);
        let anonymous_45_line_0_column_0 =
            anonymous_46_line_0_column_0.clone() * anonymous_46_line_0_column_0.clone();
        let anonymous_47_line_0_column_0 = name_imm_bytes[2usize].clone();
        let anonymous_44_line_0_column_0 =
            anonymous_45_line_0_column_0.clone() * anonymous_47_line_0_column_0.clone();
        let anonymous_38_line_0_column_0 =
            anonymous_39_line_0_column_0.clone() + anonymous_44_line_0_column_0.clone();
        let anonymous_32_line_0_column_0 = constant_expr(0);
        let temp: AB::F;
        temp = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(16);
            result
        };
        let constant_UF_16 = AB::Expr::from(temp.into());
        let anonymous_124_line_0_column_0 = constant_UF_16.clone();
        let inline_2_1_name_upper = cell(13);
        let temp: AB::F;
        temp = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(13);
            result
        };
        let constant_UF_13 = AB::Expr::from(temp.into());
        let anonymous_125_line_0_column_0 = constant_UF_13.clone();
        let anonymous_83_line_0_column_0 = name_pc_bytes[3usize].clone();
        let anonymous_84_line_0_column_0 = name_imm_bytes[2usize].clone();
        let anonymous_82_line_0_column_0 =
            anonymous_83_line_0_column_0.clone() + anonymous_84_line_0_column_0.clone();
        let anonymous_76_line_0_column_0 = name_pc_bytes[2usize].clone();
        let anonymous_77_line_0_column_0 = name_imm_bytes[1usize].clone();
        let anonymous_75_line_0_column_0 =
            anonymous_76_line_0_column_0.clone() + anonymous_77_line_0_column_0.clone();
        let name_carry1 = anonymous_66_line_0_column_0.clone();
        let anonymous_74_line_0_column_0 =
            anonymous_75_line_0_column_0.clone() + name_carry1.clone();
        let anonymous_78_line_0_column_0 = name_rd_bytes[2usize].clone();
        let anonymous_73_line_0_column_0 =
            anonymous_74_line_0_column_0.clone() - anonymous_78_line_0_column_0.clone();
        let anonymous_72_line_0_column_0 =
            anonymous_73_line_0_column_0.clone() * constant(256).inverse();
        let name_carry2 = anonymous_72_line_0_column_0.clone();
        let anonymous_81_line_0_column_0 =
            anonymous_82_line_0_column_0.clone() + name_carry2.clone();
        let temp: AB::F;
        temp = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(0 as u32);
            result
        };
        let constant_U8F_0 = AB::Expr::from(temp.into());
        let anonymous_133_line_0_column_0 = constant_U8F_0.clone();
        let anonymous_129_line_0_column_0 = name_carry2.clone() * name_carry2.clone();
        let anonymous_118_line_0_column_0 = constant_UF_1.clone();
        let anonymous_123_line_0_column_0 = constant_expr(65536);
        let anonymous_122_line_0_column_0 =
            anonymous_123_line_0_column_0.clone() * inline_2_1_name_upper.clone();
        let anonymous_128_line_0_column_0 = name_carry1.clone() * name_carry1.clone();
        let anonymous_89_line_0_column_0 = name_rd_bytes[3usize].clone();
        let anonymous_91_line_0_column_0 = name_imm_bytes[0usize].clone();
        let inline_2_name_prev_t = cell(14);
        let anonymous_126_line_0_column_0 = inline_2_name_t.clone() - inline_2_name_prev_t.clone();
        let inline_2_1_name_real_diff = anonymous_126_line_0_column_0.clone();
        let anonymous_127_line_0_column_0 =
            inline_2_1_name_real_diff.clone() - anonymous_118_line_0_column_0.clone();
        let anonymous_37_line_0_column_0 = constant_UF_1.clone();
        let anonymous_85_line_0_column_0 = name_rd_bytes[3usize].clone();
        let anonymous_80_line_0_column_0 =
            anonymous_81_line_0_column_0.clone() - anonymous_85_line_0_column_0.clone();
        let anonymous_79_line_0_column_0 =
            anonymous_80_line_0_column_0.clone() * constant(256).inverse();
        let anonymous_90_line_0_column_0 = name_pc_bytes[1usize].clone();
        let anonymous_135_line_0_column_0 = constant_U8F_0.clone();
        let anonymous_136_line_0_column_0 = constant_expr(0);
        let anonymous_27_line_0_column_0 = constant_expr(576);
        let name_imm = anonymous_38_line_0_column_0.clone();
        let anonymous_30_line_0_column_0 = constant_expr(1);
        let anonymous_31_line_0_column_0 = constant_expr(0);
        let anonymous_33_line_0_column_0 = constant_expr(0);
        let anonymous_28_line_0_column_0 = [
            name_rd_ptr.clone(),
            anonymous_29_line_0_column_0.clone(),
            name_imm.clone(),
            anonymous_30_line_0_column_0.clone(),
            anonymous_31_line_0_column_0.clone(),
            anonymous_32_line_0_column_0.clone(),
            anonymous_33_line_0_column_0.clone(),
        ];
        let anonymous_26_line_0_column_0 = [
            anonymous_27_line_0_column_0.clone(),
            anonymous_28_line_0_column_0[0usize].clone(),
            anonymous_28_line_0_column_0[1usize].clone(),
            anonymous_28_line_0_column_0[2usize].clone(),
            anonymous_28_line_0_column_0[3usize].clone(),
            anonymous_28_line_0_column_0[4usize].clone(),
            anonymous_28_line_0_column_0[5usize].clone(),
            anonymous_28_line_0_column_0[6usize].clone(),
        ];
        let temp: AB::F;
        temp = {
            let result: AB::F;
            result = AB::F::from_canonical_u32(4);
            result
        };
        let constant_UF_4 = AB::Expr::from(temp.into());
        let anonymous_22_line_0_column_0 = constant_UF_4.clone();
        let anonymous_102_line_0_column_0 = name_pc.clone() + anonymous_22_line_0_column_0.clone();
        let name_next_pc = anonymous_102_line_0_column_0.clone();
        let inline_0_name_next_t = anonymous_101_line_0_column_0.clone();
        let anonymous_139_line_0_column_0 = constant_U8F_0.clone();
        let inline_2_1_name_diff = anonymous_127_line_0_column_0.clone();
        let anonymous_131_line_0_column_0 = constant_U8F_0.clone();
        let anonymous_88_line_0_column_0 = name_rd_bytes[2usize].clone();
        let anonymous_93_line_0_column_0 = name_imm_bytes[1usize].clone();
        let anonymous_137_line_0_column_0 = constant_U8F_0.clone();
        let anonymous_138_line_0_column_0 = constant_expr(0);
        let name_instruction = [
            anonymous_26_line_0_column_0[0usize].clone(),
            anonymous_26_line_0_column_0[1usize].clone(),
            anonymous_26_line_0_column_0[2usize].clone(),
            anonymous_26_line_0_column_0[3usize].clone(),
            anonymous_26_line_0_column_0[4usize].clone(),
            anonymous_26_line_0_column_0[5usize].clone(),
            anonymous_26_line_0_column_0[6usize].clone(),
            anonymous_26_line_0_column_0[7usize].clone(),
        ];
        let anonymous_87_line_0_column_0 = name_rd_bytes[1usize].clone();
        let name_carry3 = anonymous_79_line_0_column_0.clone();
        let anonymous_130_line_0_column_0 = name_carry3.clone() * name_carry3.clone();
        let inline_2_name_prev_data = [cell(15), cell(16), cell(17), cell(18)];
        let anonymous_132_line_0_column_0 = constant_expr(0);
        let anonymous_121_line_0_column_0 =
            inline_2_1_name_diff.clone() - anonymous_122_line_0_column_0.clone();
        let inline_2_1_name_lower = anonymous_121_line_0_column_0.clone();
        let anonymous_95_line_0_column_0 = name_imm_bytes[2usize].clone();
        let anonymous_134_line_0_column_0 = constant_expr(0);
        let anonymous_94_line_0_column_0 = name_pc_bytes[3usize].clone();
        "line 33, column 17: x * x == x";
        builder
            .when(scope.clone())
            .assert_eq(anonymous_128_line_0_column_0.clone(), name_carry1.clone());
        "line 33, column 17: x * x == x";
        builder
            .when(scope.clone())
            .assert_eq(anonymous_130_line_0_column_0.clone(), name_carry3.clone());
        "line 33, column 17: x * x == x";
        builder
            .when(scope.clone())
            .assert_eq(anonymous_129_line_0_column_0.clone(), name_carry2.clone());
        "line 8, column 17: range_check << x, num_bits";
        builder.push_interaction(
            self.custom_bus_range_check,
            [
                inline_2_1_name_upper.clone(),
                anonymous_125_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 41, column 9: memory << address_space, pointer, data, t";
        builder.push_interaction(
            self.custom_bus_memory,
            [
                anonymous_37_line_0_column_0.clone(),
                name_rd_ptr.clone(),
                name_rd_bytes[0usize].clone(),
                name_rd_bytes[1usize].clone(),
                name_rd_bytes[2usize].clone(),
                name_rd_bytes[3usize].clone(),
                inline_2_name_t.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 8, column 17: bitwise << x, y, 0U8F, 0";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [
                anonymous_90_line_0_column_0.clone(),
                anonymous_91_line_0_column_0.clone(),
                anonymous_135_line_0_column_0.clone(),
                anonymous_136_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 75, column 9: exe << next_pc, next_t";
        builder.push_interaction(
            self.custom_bus_exe,
            [name_next_pc.clone(), inline_0_name_next_t.clone()],
            scope.clone() * constant(1),
            1,
        );
        "line 8, column 17: bitwise << x, y, 0U8F, 0";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [
                anonymous_92_line_0_column_0.clone(),
                anonymous_93_line_0_column_0.clone(),
                anonymous_137_line_0_column_0.clone(),
                anonymous_138_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 76, column 9: program << start_pc, instruction";
        builder.push_interaction(
            self.custom_bus_program,
            [
                name_pc.clone(),
                name_instruction[0usize].clone(),
                name_instruction[1usize].clone(),
                name_instruction[2usize].clone(),
                name_instruction[3usize].clone(),
                name_instruction[4usize].clone(),
                name_instruction[5usize].clone(),
                name_instruction[6usize].clone(),
                name_instruction[7usize].clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 8, column 17: bitwise << x, y, 0U8F, 0";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [
                anonymous_86_line_0_column_0.clone(),
                anonymous_87_line_0_column_0.clone(),
                anonymous_131_line_0_column_0.clone(),
                anonymous_132_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 74, column 9: exe >> start_pc, start_t";
        builder.push_interaction(
            self.custom_bus_exe,
            [name_pc.clone(), inline_0_name_start_t.clone()],
            -(scope.clone() * constant(1)),
            1,
        );
        "line 8, column 17: range_check << x, num_bits";
        builder.push_interaction(
            self.custom_bus_range_check,
            [
                inline_2_1_name_lower.clone(),
                anonymous_124_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 40, column 9: memory >> address_space, pointer, prev_data, prev_t";
        builder.push_interaction(
            self.custom_bus_memory,
            [
                anonymous_37_line_0_column_0.clone(),
                name_rd_ptr.clone(),
                inline_2_name_prev_data[0usize].clone(),
                inline_2_name_prev_data[1usize].clone(),
                inline_2_name_prev_data[2usize].clone(),
                inline_2_name_prev_data[3usize].clone(),
                inline_2_name_prev_t.clone(),
            ],
            -(scope.clone() * constant(1)),
            1,
        );
        "line 8, column 17: bitwise << x, y, 0U8F, 0";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [
                anonymous_94_line_0_column_0.clone(),
                anonymous_95_line_0_column_0.clone(),
                anonymous_139_line_0_column_0.clone(),
                anonymous_140_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
        "line 8, column 17: bitwise << x, y, 0U8F, 0";
        builder.push_interaction(
            self.custom_bus_bitwise,
            [
                anonymous_88_line_0_column_0.clone(),
                anonymous_89_line_0_column_0.clone(),
                anonymous_133_line_0_column_0.clone(),
                anonymous_134_line_0_column_0.clone(),
            ],
            scope.clone() * constant(1),
            1,
        );
    }
}
#[derive(Clone, derive_new :: new)]
pub struct Rv32AuipcDazlStep {}
impl<F, RA> InstructionExecutor<F, RA> for Rv32AuipcDazlStep
where
    F: PrimeField32,
    for<'buf> RA:
    RecordArena<'buf, MultiRowLayout<EmptyMultiRowMetadata>, &'buf mut Rv32AuipcDazlRecord>,
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
        let mut name_rd: u32 = Default::default();
        let mut unique_121_0: usize = Default::default();
        let mut name_opcode: u32 = Default::default();
        let mut unique_114_0: usize = Default::default();
        let mut anonymous_89_line_0_column_0: u8 = Default::default();
        let mut unique_143_0: usize = Default::default();
        let mut unique_128_0: usize = Default::default();
        let mut unique_136_0: usize = Default::default();
        let mut name_real_imm: u32 = Default::default();
        let mut unique_97_0: usize = Default::default();
        let mut anonymous_24_line_0_column_0: u32 = Default::default();
        let mut anonymous_25_line_0_column_0: u32 = Default::default();
        let mut unique_24_0: usize = Default::default();
        let mut anonymous_95_line_0_column_0: u8 = Default::default();
        let mut unique_167_0: usize = Default::default();
        let mut anonymous_94_line_0_column_0: u8 = Default::default();
        let mut name_imm_bytes: [u8; 3usize] = Default::default();
        let mut unique_139_0: usize = Default::default();
        let mut name_end_baton: () = Default::default();
        let mut name_rd_ptr: u32 = Default::default();
        let mut name_start_baton: () = Default::default();
        let mut unique_22_0: usize = Default::default();
        let mut anonymous_101_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_120_line_0_column_0: u32 = Default::default();
        let mut anonymous_35_line_0_column_0: [u8; 1usize] = Default::default();
        let mut unique_18_0: usize = Default::default();
        let mut call_index: usize = Default::default();
        let mut unique_106_0: usize = Default::default();
        let mut inline_0_name_baton2: () = Default::default();
        let mut name_operands: [u32; 7usize] = Default::default();
        let mut anonymous_92_line_0_column_0: u8 = Default::default();
        let mut anonymous_109_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_86_line_0_column_0: u8 = Default::default();
        let mut anonymous_119_line_0_column_0: u32 = Default::default();
        let mut name_end_timestamp: TL_Timestamp = Default::default();
        let mut unique_25_0: usize = Default::default();
        let mut inline_2_name_baton0: () = Default::default();
        let mut unique_159_0: usize = Default::default();
        let mut unique_102_0: usize = Default::default();
        let mut inline_0_name_next_t: u32 = Default::default();
        let mut anonymous_88_line_0_column_0: u8 = Default::default();
        let mut anonymous_36_line_0_column_0: u32 = Default::default();
        let mut name_rd_bytes: [u8; 4usize] = Default::default();
        let mut anonymous_93_line_0_column_0: u8 = Default::default();
        let mut anonymous_22_line_0_column_0: u32 = Default::default();
        let mut anonymous_118_line_0_column_0: u32 = Default::default();
        let mut unique_145_0: usize = Default::default();
        let mut anonymous_90_line_0_column_0: u8 = Default::default();
        let mut anonymous_91_line_0_column_0: u8 = Default::default();
        let mut unique_155_0: usize = Default::default();
        let mut unique_38_0: usize = Default::default();
        let mut name_start_timestamp: TL_Timestamp = Default::default();
        let mut anonymous_23_line_0_column_0: TL_Instruction = Default::default();
        let mut inline_2_name_t: u32 = Default::default();
        let mut anonymous_100_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_111_line_0_column_0: u32 = Default::default();
        let mut unique_163_0: usize = Default::default();
        let mut anonymous_110_line_0_column_0: u32 = Default::default();
        let mut inline_2_1_name_upper: u32 = Default::default();
        let mut unique_171_0: usize = Default::default();
        let mut anonymous_112_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_125_line_0_column_0: u32 = Default::default();
        let mut anonymous_124_line_0_column_0: u32 = Default::default();
        let mut anonymous_87_line_0_column_0: u8 = Default::default();
        let mut anonymous_34_line_0_column_0: [u8; 4usize] = Default::default();
        let mut unique_126_0: usize = Default::default();
        name_start_baton = argument_0;
        "line 68, column 15: execution_start(start_baton, start_pc, instruction, start_t, baton1)";
        record.name_pc = *vm_state.pc;
        record.name_instruction = TL_Instruction::Instruction(
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
        record.inline_0_name_start_t = vm_state.memory.timestamp;
        "line 69, column 23: Timestamp(baton1, start_t)";
        anonymous_100_line_0_column_0 =
            TL_Timestamp::Timestamp(record.inline_0_name_baton1, record.inline_0_name_start_t);
        "line 69, column 5: start_timestamp = Timestamp(baton1, start_t)";
        name_start_timestamp = anonymous_100_line_0_column_0;
        "line 26, column 5: Timestamp(let baton0, let t) = timestamp";
        anonymous_109_line_0_column_0 = name_start_timestamp;
        "line 26, column 5: Timestamp(let baton0, let t)";
        match anonymous_109_line_0_column_0 {
            TL_Timestamp::Timestamp(temp_0, temp_1) => {
                inline_2_name_baton0 = temp_0;
                inline_2_name_t = temp_1;
            }
            _ => panic!("unexpected constructor"),
        }
        "line 34, column 38: 1UF";
        anonymous_110_line_0_column_0 = (1 as u32);
        "line 17, column 9: Instruction(let opcode, let operands) = instruction";
        anonymous_23_line_0_column_0 = record.name_instruction;
        "line 17, column 9: Instruction(let opcode, let operands)";
        match anonymous_23_line_0_column_0 {
            TL_Instruction::Instruction(temp_0, temp_1) => {
                name_opcode = temp_0;
                name_operands = temp_1;
            }
            _ => panic!("unexpected constructor"),
        }
        "line 18, column 18: operands[0]";
        anonymous_24_line_0_column_0 = name_operands[0];
        "line 18, column 9: rd_ptr = operands[0]";
        name_rd_ptr = anonymous_24_line_0_column_0;
        "line 19, column 15: operands[1]";
        anonymous_25_line_0_column_0 = name_operands[1];
        "line 19, column 9: imm = operands[1]";
        record.name_imm = anonymous_25_line_0_column_0;
        "line 33, column 21: 8U32";
        anonymous_36_line_0_column_0 = (8 as u32);
        "line 33, column 9: lshift(imm, 8U32, let real_imm)";
        name_real_imm = record.name_imm << anonymous_36_line_0_column_0;
        "line 34, column 9: uadd(pc, real_imm, let rd)";
        name_rd = record.name_pc.wrapping_add(name_real_imm);
        "line 35, column 9: u32_to_bytes(rd, rd_bytes)";
        name_rd_bytes = name_rd.to_le_bytes();
        "line 34, column 15: e_write_memory(baton0, 1UF, pointer, data, prev_t, prev_data, baton1)";
        tracing_write(
            vm_state.memory,
            anonymous_110_line_0_column_0,
            name_rd_ptr,
            name_rd_bytes,
            &mut record.inline_2_name_prev_t,
            &mut record.inline_2_name_prev_data,
        );
        "line 36, column 15: 1UF";
        anonymous_111_line_0_column_0 = (1 as u32);
        "line 10, column 15: uadd(x, y, z)";
        record.inline_2_name_next_t = inline_2_name_t.wrapping_add(anonymous_111_line_0_column_0);
        "line 37, column 22: Timestamp(baton1, next_t)";
        anonymous_112_line_0_column_0 =
            TL_Timestamp::Timestamp(record.inline_2_name_baton1, record.inline_2_name_next_t);
        "line 37, column 5: next_timestamp = Timestamp(baton1, next_t)";
        name_end_timestamp = anonymous_112_line_0_column_0;
        "line 70, column 5: Timestamp(let baton2, let next_t) = next_timestamp";
        anonymous_101_line_0_column_0 = name_end_timestamp;
        "line 70, column 5: Timestamp(let baton2, let next_t)";
        match anonymous_101_line_0_column_0 {
            TL_Timestamp::Timestamp(temp_0, temp_1) => {
                inline_0_name_baton2 = temp_0;
                inline_0_name_next_t = temp_1;
            }
            _ => panic!("unexpected constructor"),
        }
        "line 11, column 16: 4UF";
        anonymous_22_line_0_column_0 = (4 as u32);
        "line 10, column 15: uadd(x, y, z)";
        record.name_next_pc = record.name_pc.wrapping_add(anonymous_22_line_0_column_0);
        "line 71, column 15: execution_end(baton2, next_pc, end_baton)";
        *vm_state.pc = record.name_next_pc;
        "line 15, column 15: usub(x, y, z)";
        record.inline_2_1_name_real_diff =
            inline_2_name_t.wrapping_sub(record.inline_2_name_prev_t);
        "line 14, column 23: 1UF";
        anonymous_118_line_0_column_0 = (1 as u32);
        "line 15, column 15: usub(x, y, z)";
        record.inline_2_1_name_diff = record
            .inline_2_1_name_real_diff
            .wrapping_sub(anonymous_118_line_0_column_0);
        "line 21, column 27: 65535U32";
        anonymous_119_line_0_column_0 = (65535 as u32);
        "line 21, column 9: bitwise_and(diff, 65535U32, lower)";
        record.inline_2_1_name_lower = record.inline_2_1_name_diff & anonymous_119_line_0_column_0;
        "line 26, column 24: 16UF";
        anonymous_124_line_0_column_0 = (16 as u32);
        "line 9, column 15: balance_range_check(x, num_bits)";
        "line 22, column 22: 16U32";
        anonymous_120_line_0_column_0 = (16 as u32);
        "line 22, column 9: rshift(diff, 16U32, upper)";
        inline_2_1_name_upper = record.inline_2_1_name_diff >> anonymous_120_line_0_column_0;
        "line 27, column 24: 13UF";
        anonymous_125_line_0_column_0 = (13 as u32);
        "line 9, column 15: balance_range_check(x, num_bits)";
        "line 61, column 25: rd_bytes[0]";
        anonymous_86_line_0_column_0 = name_rd_bytes[0];
        "line 61, column 38: rd_bytes[1]";
        anonymous_87_line_0_column_0 = name_rd_bytes[1];
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        "line 62, column 25: rd_bytes[2]";
        anonymous_88_line_0_column_0 = name_rd_bytes[2];
        "line 62, column 38: rd_bytes[3]";
        anonymous_89_line_0_column_0 = name_rd_bytes[3];
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        "line 43, column 15: u32_to_bytes(pc, pc_bytes)";
        record.name_pc_bytes = record.name_pc.to_le_bytes();
        "line 63, column 25: pc_bytes[1]";
        anonymous_90_line_0_column_0 = record.name_pc_bytes[1];
        "line 27, column 9: u32_to_bytes(imm, imm_bytes ++ [unused])";
        anonymous_34_line_0_column_0 = record.name_imm.to_le_bytes();
        "line 27, column 27: imm_bytes ++ [unused]";
        name_imm_bytes = anonymous_34_line_0_column_0[..3].try_into().unwrap();
        anonymous_35_line_0_column_0 = anonymous_34_line_0_column_0[3..4].try_into().unwrap();
        "line 63, column 38: imm_bytes[0]";
        anonymous_91_line_0_column_0 = name_imm_bytes[0];
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        "line 64, column 25: pc_bytes[2]";
        anonymous_92_line_0_column_0 = record.name_pc_bytes[2];
        "line 64, column 38: imm_bytes[1]";
        anonymous_93_line_0_column_0 = name_imm_bytes[1];
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        "line 65, column 25: pc_bytes[3]";
        anonymous_94_line_0_column_0 = record.name_pc_bytes[3];
        "line 65, column 38: imm_bytes[2]";
        anonymous_95_line_0_column_0 = name_imm_bytes[2];
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        (name_end_baton);
        Ok(())
    }
}
#[derive(Clone, derive_new :: new)]
pub struct Rv32AuipcDazlFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}
impl<F: PrimeField32> TraceFiller<F> for Rv32AuipcDazlFiller {
    fn fill_trace_row(&self, _: &MemoryAuxColsFactory<F>, mut row_slice: &mut [F]) {
        let record: &Rv32AuipcDazlRecord = unsafe { get_record_from_slice(&mut row_slice, ()) };
        let mut record = record.clone();
        let mut name_rd: u32 = Default::default();
        let mut unique_121_0: usize = Default::default();
        let mut anonymous_125_line_0_column_0: u32 = Default::default();
        let mut name_opcode: u32 = Default::default();
        let mut anonymous_89_line_0_column_0: u8 = Default::default();
        let mut unique_128_0: usize = Default::default();
        let mut unique_136_0: usize = Default::default();
        let mut name_real_imm: u32 = Default::default();
        let mut anonymous_24_line_0_column_0: u32 = Default::default();
        let mut name_pc_middle: [u8; 2usize] = Default::default();
        let mut anonymous_25_line_0_column_0: u32 = Default::default();
        let mut unique_24_0: usize = Default::default();
        let mut anonymous_95_line_0_column_0: u8 = Default::default();
        let mut anonymous_94_line_0_column_0: u8 = Default::default();
        let mut name_imm_bytes: [u8; 3usize] = Default::default();
        let mut unique_139_0: usize = Default::default();
        let mut name_rd_ptr: u32 = Default::default();
        let mut unique_22_0: usize = Default::default();
        let mut anonymous_101_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_120_line_0_column_0: u32 = Default::default();
        let mut anonymous_35_line_0_column_0: [u8; 1usize] = Default::default();
        let mut unique_18_0: usize = Default::default();
        let mut unique_106_0: usize = Default::default();
        let mut inline_0_name_baton2: () = Default::default();
        let mut name_operands: [u32; 7usize] = Default::default();
        let mut anonymous_92_line_0_column_0: u8 = Default::default();
        let mut anonymous_109_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_119_line_0_column_0: u32 = Default::default();
        let mut anonymous_86_line_0_column_0: u8 = Default::default();
        let mut name_end_timestamp: TL_Timestamp = Default::default();
        let mut unique_25_0: usize = Default::default();
        let mut inline_2_name_baton0: () = Default::default();
        let mut inline_0_name_next_t: u32 = Default::default();
        let mut anonymous_88_line_0_column_0: u8 = Default::default();
        let mut anonymous_36_line_0_column_0: u32 = Default::default();
        let mut name_rd_bytes: [u8; 4usize] = Default::default();
        let mut anonymous_22_line_0_column_0: u32 = Default::default();
        let mut anonymous_93_line_0_column_0: u8 = Default::default();
        let mut anonymous_118_line_0_column_0: u32 = Default::default();
        let mut name_unused: u8 = Default::default();
        let mut anonymous_90_line_0_column_0: u8 = Default::default();
        let mut anonymous_91_line_0_column_0: u8 = Default::default();
        let mut unique_38_0: usize = Default::default();
        let mut name_start_timestamp: TL_Timestamp = Default::default();
        let mut anonymous_23_line_0_column_0: TL_Instruction = Default::default();
        let mut inline_2_name_t: u32 = Default::default();
        let mut anonymous_100_line_0_column_0: TL_Timestamp = Default::default();
        let mut anonymous_111_line_0_column_0: u32 = Default::default();
        let mut anonymous_48_line_0_column_0: [u8; 2usize] = Default::default();
        let mut anonymous_110_line_0_column_0: u32 = Default::default();
        let mut inline_2_1_name_upper: u32 = Default::default();
        let mut anonymous_112_line_0_column_0: TL_Timestamp = Default::default();
        let mut unique_126_0: usize = Default::default();
        let mut anonymous_124_line_0_column_0: u32 = Default::default();
        let mut anonymous_87_line_0_column_0: u8 = Default::default();
        let mut anonymous_37_line_0_column_0: u32 = Default::default();
        let mut anonymous_34_line_0_column_0: [u8; 4usize] = Default::default();
        "line 69, column 23: Timestamp(baton1, start_t)";
        anonymous_100_line_0_column_0 =
            TL_Timestamp::Timestamp(record.inline_0_name_baton1, record.inline_0_name_start_t);
        "line 69, column 5: start_timestamp = Timestamp(baton1, start_t)";
        name_start_timestamp = anonymous_100_line_0_column_0;
        "line 26, column 5: Timestamp(let baton0, let t) = timestamp";
        anonymous_109_line_0_column_0 = name_start_timestamp;
        "line 26, column 5: Timestamp(let baton0, let t)";
        match anonymous_109_line_0_column_0 {
            TL_Timestamp::Timestamp(temp_0, temp_1) => {
                inline_2_name_baton0 = temp_0;
                inline_2_name_t = temp_1;
            }
            _ => panic!("unexpected constructor"),
        }
        "line 34, column 38: 1UF";
        anonymous_110_line_0_column_0 = (1 as u32);
        "line 17, column 9: Instruction(let opcode, let operands) = instruction";
        anonymous_23_line_0_column_0 = record.name_instruction;
        "line 17, column 9: Instruction(let opcode, let operands)";
        match anonymous_23_line_0_column_0 {
            TL_Instruction::Instruction(temp_0, temp_1) => {
                name_opcode = temp_0;
                name_operands = temp_1;
            }
            _ => panic!("unexpected constructor"),
        }
        "line 18, column 18: operands[0]";
        anonymous_24_line_0_column_0 = name_operands[0];
        "line 18, column 9: rd_ptr = operands[0]";
        name_rd_ptr = anonymous_24_line_0_column_0;
        "line 19, column 15: operands[1]";
        anonymous_25_line_0_column_0 = name_operands[1];
        "line 19, column 9: imm = operands[1]";
        record.name_imm = anonymous_25_line_0_column_0;
        "line 33, column 21: 8U32";
        anonymous_36_line_0_column_0 = (8 as u32);
        "line 33, column 9: lshift(imm, 8U32, let real_imm)";
        name_real_imm = record.name_imm << anonymous_36_line_0_column_0;
        "line 34, column 9: uadd(pc, real_imm, let rd)";
        name_rd = record.name_pc.wrapping_add(name_real_imm);
        "line 35, column 9: u32_to_bytes(rd, rd_bytes)";
        name_rd_bytes = name_rd.to_le_bytes();
        "line 36, column 15: 1UF";
        anonymous_111_line_0_column_0 = (1 as u32);
        "line 10, column 15: uadd(x, y, z)";
        record.inline_2_name_next_t = inline_2_name_t.wrapping_add(anonymous_111_line_0_column_0);
        "line 37, column 22: Timestamp(baton1, next_t)";
        anonymous_112_line_0_column_0 =
            TL_Timestamp::Timestamp(record.inline_2_name_baton1, record.inline_2_name_next_t);
        "line 37, column 5: next_timestamp = Timestamp(baton1, next_t)";
        name_end_timestamp = anonymous_112_line_0_column_0;
        "line 70, column 5: Timestamp(let baton2, let next_t) = next_timestamp";
        anonymous_101_line_0_column_0 = name_end_timestamp;
        "line 70, column 5: Timestamp(let baton2, let next_t)";
        match anonymous_101_line_0_column_0 {
            TL_Timestamp::Timestamp(temp_0, temp_1) => {
                inline_0_name_baton2 = temp_0;
                inline_0_name_next_t = temp_1;
            }
            _ => panic!("unexpected constructor"),
        }
        "line 11, column 16: 4UF";
        anonymous_22_line_0_column_0 = (4 as u32);
        "line 10, column 15: uadd(x, y, z)";
        record.name_next_pc = record.name_pc.wrapping_add(anonymous_22_line_0_column_0);
        "line 15, column 15: usub(x, y, z)";
        record.inline_2_1_name_real_diff =
            inline_2_name_t.wrapping_sub(record.inline_2_name_prev_t);
        "line 14, column 23: 1UF";
        anonymous_118_line_0_column_0 = (1 as u32);
        "line 15, column 15: usub(x, y, z)";
        record.inline_2_1_name_diff = record
            .inline_2_1_name_real_diff
            .wrapping_sub(anonymous_118_line_0_column_0);
        "line 21, column 27: 65535U32";
        anonymous_119_line_0_column_0 = (65535 as u32);
        "line 21, column 9: bitwise_and(diff, 65535U32, lower)";
        record.inline_2_1_name_lower = record.inline_2_1_name_diff & anonymous_119_line_0_column_0;
        "line 26, column 24: 16UF";
        anonymous_124_line_0_column_0 = (16 as u32);
        "line 22, column 22: 16U32";
        anonymous_120_line_0_column_0 = (16 as u32);
        "line 22, column 9: rshift(diff, 16U32, upper)";
        inline_2_1_name_upper = record.inline_2_1_name_diff >> anonymous_120_line_0_column_0;
        "line 27, column 24: 13UF";
        anonymous_125_line_0_column_0 = (13 as u32);
        "line 61, column 25: rd_bytes[0]";
        anonymous_86_line_0_column_0 = name_rd_bytes[0];
        "line 61, column 38: rd_bytes[1]";
        anonymous_87_line_0_column_0 = name_rd_bytes[1];
        "line 62, column 25: rd_bytes[2]";
        anonymous_88_line_0_column_0 = name_rd_bytes[2];
        "line 62, column 38: rd_bytes[3]";
        anonymous_89_line_0_column_0 = name_rd_bytes[3];
        "line 43, column 15: u32_to_bytes(pc, pc_bytes)";
        record.name_pc_bytes = record.name_pc.to_le_bytes();
        "line 63, column 25: pc_bytes[1]";
        anonymous_90_line_0_column_0 = record.name_pc_bytes[1];
        "line 27, column 9: u32_to_bytes(imm, imm_bytes ++ [unused])";
        anonymous_34_line_0_column_0 = record.name_imm.to_le_bytes();
        "line 27, column 27: imm_bytes ++ [unused]";
        name_imm_bytes = anonymous_34_line_0_column_0[..3].try_into().unwrap();
        anonymous_35_line_0_column_0 = anonymous_34_line_0_column_0[3..4].try_into().unwrap();
        "line 63, column 38: imm_bytes[0]";
        anonymous_91_line_0_column_0 = name_imm_bytes[0];
        "line 64, column 25: pc_bytes[2]";
        anonymous_92_line_0_column_0 = record.name_pc_bytes[2];
        "line 64, column 38: imm_bytes[1]";
        anonymous_93_line_0_column_0 = name_imm_bytes[1];
        "line 65, column 25: pc_bytes[3]";
        anonymous_94_line_0_column_0 = record.name_pc_bytes[3];
        "line 65, column 38: imm_bytes[2]";
        anonymous_95_line_0_column_0 = name_imm_bytes[2];
        "line 31, column 5: alloc baton1";
        "line 30, column 5: alloc prev_data";
        "line 31, column 5: alloc rd_bytes";
        "line 15, column 5: alloc rd_ptr";
        "line 66, column 5: alloc start_t";
        "line 46, column 27: pc_bytes[1..3]";
        anonymous_48_line_0_column_0 = record.name_pc_bytes[1..3].try_into().unwrap();
        "line 46, column 15: pc_middle = pc_bytes[1..3]";
        name_pc_middle = anonymous_48_line_0_column_0;
        "line 38, column 35: 1UF";
        anonymous_37_line_0_column_0 = (1 as u32);
        "line 24, column 5: alloc imm_bytes";
        "line 65, column 5: alloc baton1";
        "line 18, column 5: alloc upper";
        "line 45, column 5: alloc pc_middle";
        "line 32, column 5: alloc prev_t";
        "line 27, column 40: [unused]";
        name_unused = anonymous_35_line_0_column_0[0];
        "line 62, column 5: alloc start_pc";
        row_slice[0usize] = F::ONE;
        "line 15, column 5: alloc rd_ptr";
        row_slice[12usize] = {
            let result: F;
            result = F::from_canonical_u32(name_rd_ptr);
            result
        };
        "line 24, column 5: alloc imm_bytes";
        row_slice[6usize] = {
            let result: F;
            result = F::from_canonical_u32(name_imm_bytes[0usize] as u32);
            result
        };
        row_slice[7usize] = {
            let result: F;
            result = F::from_canonical_u32(name_imm_bytes[1usize] as u32);
            result
        };
        row_slice[8usize] = {
            let result: F;
            result = F::from_canonical_u32(name_imm_bytes[2usize] as u32);
            result
        };
        "line 31, column 5: alloc rd_bytes";
        row_slice[2usize] = {
            let result: F;
            result = F::from_canonical_u32(name_rd_bytes[0usize] as u32);
            result
        };
        row_slice[3usize] = {
            let result: F;
            result = F::from_canonical_u32(name_rd_bytes[1usize] as u32);
            result
        };
        row_slice[4usize] = {
            let result: F;
            result = F::from_canonical_u32(name_rd_bytes[2usize] as u32);
            result
        };
        row_slice[5usize] = {
            let result: F;
            result = F::from_canonical_u32(name_rd_bytes[3usize] as u32);
            result
        };
        "line 45, column 5: alloc pc_middle";
        row_slice[9usize] = {
            let result: F;
            result = F::from_canonical_u32(name_pc_middle[0usize] as u32);
            result
        };
        row_slice[10usize] = {
            let result: F;
            result = F::from_canonical_u32(name_pc_middle[1usize] as u32);
            result
        };
        "line 62, column 5: alloc start_pc";
        row_slice[11usize] = {
            let result: F;
            result = F::from_canonical_u32(record.name_pc);
            result
        };
        "line 65, column 5: alloc baton1";
        "line 66, column 5: alloc start_t";
        row_slice[1usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline_0_name_start_t);
            result
        };
        "line 30, column 5: alloc prev_data";
        row_slice[15usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline_2_name_prev_data[0usize] as u32);
            result
        };
        row_slice[16usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline_2_name_prev_data[1usize] as u32);
            result
        };
        row_slice[17usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline_2_name_prev_data[2usize] as u32);
            result
        };
        row_slice[18usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline_2_name_prev_data[3usize] as u32);
            result
        };
        "line 31, column 5: alloc baton1";
        "line 32, column 5: alloc prev_t";
        row_slice[14usize] = {
            let result: F;
            result = F::from_canonical_u32(record.inline_2_name_prev_t);
            result
        };
        "line 18, column 5: alloc upper";
        row_slice[13usize] = {
            let result: F;
            result = F::from_canonical_u32(inline_2_1_name_upper);
            result
        };
        "line 9, column 15: balance_range_check(x, num_bits)";
        self.range_checker_chip.add_count(
            record.inline_2_1_name_lower,
            anonymous_124_line_0_column_0 as usize,
        );
        "line 9, column 15: balance_range_check(x, num_bits)";
        self.range_checker_chip.add_count(
            inline_2_1_name_upper,
            anonymous_125_line_0_column_0 as usize,
        );
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip.request_range(
            anonymous_86_line_0_column_0 as u32,
            anonymous_87_line_0_column_0 as u32,
        );
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip.request_range(
            anonymous_88_line_0_column_0 as u32,
            anonymous_89_line_0_column_0 as u32,
        );
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip.request_range(
            anonymous_90_line_0_column_0 as u32,
            anonymous_91_line_0_column_0 as u32,
        );
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip.request_range(
            anonymous_92_line_0_column_0 as u32,
            anonymous_93_line_0_column_0 as u32,
        );
        "line 9, column 15: balance_bitwise_range_check(x, y)";
        self.bitwise_lookup_chip.request_range(
            anonymous_94_line_0_column_0 as u32,
            anonymous_95_line_0_column_0 as u32,
        );
    }
}
