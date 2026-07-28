use std::{array, borrow::BorrowMut, iter::once};

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        fill_trace_rows, Postflight, PostflightError, PostflightStep, BLOCK_FE_WIDTH,
        MEMORY_BLOCK_BYTES,
    },
    system::memory::MemoryAuxColsFactory,
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, VmOpcode,
};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterCols, Rv64VecHeapBranchU16AdapterCols, Rv64VecHeapU16AdapterCols,
};
use openvm_riscv_circuit::{
    adapters::{ptr_bound_from_ptr, ptr_to_u16_limbs, U16_BITS},
    AddSubCoreCols, BitwiseLogicCoreCols, BranchEqualCoreCols, BranchLessThanCoreCols,
    LessThanCoreCols, MultiplicationCoreCols, ShiftLogicalCoreCols, ShiftRightArithmeticCoreCols,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use crate::{
    mult::u256_mul, Rv64AddSub256Chip, Rv64BitwiseLogic256Chip, Rv64BranchEqual256Chip,
    Rv64BranchLessThan256Chip, Rv64LessThan256Chip, Rv64Multiplication256Chip,
    Rv64ShiftLogical256Chip, Rv64ShiftRightArithmetic256Chip, INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_U16_LIMBS, INT256_NUM_U8_LIMBS, NUM_READS, RV64_BYTE_BITS,
};

type AluU16Cols<F> =
    Rv64VecHeapU16AdapterCols<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
type AluByteCols<F> =
    Rv64VecHeapAdapterCols<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
type BranchCols<F> = Rv64VecHeapBranchU16AdapterCols<F, NUM_READS, INT256_NUM_MEMORY_BLOCKS>;

struct AluReplay<T, const NUM_LIMBS: usize> {
    inputs: [[T; NUM_LIMBS]; NUM_READS],
    output: [T; NUM_LIMBS],
}

fn invalid(message: impl Into<String>) -> PostflightError {
    PostflightError::new(message)
}

fn checked_register_pointer(byte_pointer: u32) -> Result<u32, PostflightError> {
    if !byte_pointer.is_multiple_of(2) {
        return Err(invalid("register byte pointer must be two-byte aligned"));
    }
    Ok(byte_pointer / 2)
}

fn decode_heap_pointer(
    value: [u16; BLOCK_FE_WIDTH],
    pointer_max_bits: usize,
) -> Result<u32, PostflightError> {
    if value[2..] != [0, 0] {
        return Err(invalid("heap pointer register has nonzero upper cells"));
    }
    let pointer = u32::from(value[0]) | (u32::from(value[1]) << U16_BITS);
    if pointer_max_bits < u32::BITS as usize && pointer >= 1u32 << pointer_max_bits {
        return Err(invalid("heap pointer exceeds configured pointer width"));
    }
    Ok(pointer)
}

fn validate_heap_span(pointer: u32, pointer_max_bits: usize) -> Result<(), PostflightError> {
    if !pointer.is_multiple_of(MEMORY_BLOCK_BYTES as u32) {
        return Err(invalid("heap pointer must be memory-block aligned"));
    }
    let last = pointer
        .checked_add(INT256_NUM_U8_LIMBS as u32 - 1)
        .ok_or_else(|| invalid("heap access overflows u32"))?;
    if pointer_max_bits < u32::BITS as usize && last >= 1u32 << pointer_max_bits {
        return Err(invalid("heap access exceeds configured pointer width"));
    }
    Ok(())
}

fn validate_alu_instruction<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> Result<(), PostflightError> {
    if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
        || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
    {
        return Err(invalid("int256 ALU instruction has invalid address spaces"));
    }
    Ok(())
}

fn read_pointer_register<F: PrimeField32>(
    replay: &mut openvm_circuit::arch::PostflightReplay<'_, '_, F>,
    byte_pointer: u32,
    pointer_max_bits: usize,
) -> Result<(openvm_circuit::arch::U16Access, u32), PostflightError> {
    let access = replay.read_u16(RV64_REGISTER_AS, checked_register_pointer(byte_pointer)?)?;
    let pointer = decode_heap_pointer(access.value, pointer_max_bits)?;
    validate_heap_span(pointer, pointer_max_bits)?;
    Ok((access, pointer))
}

fn u16_block_to_bytes(block: [u16; BLOCK_FE_WIDTH]) -> [u8; MEMORY_BLOCK_BYTES] {
    array::from_fn(|i| block[i / 2].to_le_bytes()[i % 2])
}

fn bytes_to_u16_block(block: [u8; MEMORY_BLOCK_BYTES]) -> [u16; BLOCK_FE_WIDTH] {
    array::from_fn(|i| u16::from_le_bytes([block[2 * i], block[2 * i + 1]]))
}

fn flatten_u16_blocks(
    blocks: [[u16; BLOCK_FE_WIDTH]; INT256_NUM_MEMORY_BLOCKS],
) -> [u16; INT256_NUM_U16_LIMBS] {
    array::from_fn(|i| blocks[i / BLOCK_FE_WIDTH][i % BLOCK_FE_WIDTH])
}

fn split_u16_blocks(
    limbs: [u16; INT256_NUM_U16_LIMBS],
) -> [[u16; BLOCK_FE_WIDTH]; INT256_NUM_MEMORY_BLOCKS] {
    array::from_fn(|i| array::from_fn(|j| limbs[i * BLOCK_FE_WIDTH + j]))
}

fn flatten_byte_blocks(
    blocks: [[u8; MEMORY_BLOCK_BYTES]; INT256_NUM_MEMORY_BLOCKS],
) -> [u8; INT256_NUM_U8_LIMBS] {
    array::from_fn(|i| blocks[i / MEMORY_BLOCK_BYTES][i % MEMORY_BLOCK_BYTES])
}

fn split_byte_blocks(
    bytes: [u8; INT256_NUM_U8_LIMBS],
) -> [[u8; MEMORY_BLOCK_BYTES]; INT256_NUM_MEMORY_BLOCKS] {
    array::from_fn(|i| array::from_fn(|j| bytes[i * MEMORY_BLOCK_BYTES + j]))
}

fn replay_alu_u16<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
    mem_helper: &MemoryAuxColsFactory<F>,
    range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    adapter_row: &mut AluU16Cols<F>,
    compute: impl FnOnce([[u16; INT256_NUM_U16_LIMBS]; NUM_READS]) -> [u16; INT256_NUM_U16_LIMBS],
) -> Result<AluReplay<u16, INT256_NUM_U16_LIMBS>, PostflightError> {
    let instruction = postflight.instruction(step);
    validate_alu_instruction(instruction)?;
    let rs_ptrs = [
        instruction.b.as_canonical_u32(),
        instruction.c.as_canonical_u32(),
    ];
    let rd_ptr = instruction.a.as_canonical_u32();
    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let mut replay = postflight.replay(step);

    let mut rs_accesses = Vec::with_capacity(NUM_READS);
    let mut rs_vals = [0u32; NUM_READS];
    for (i, &pointer) in rs_ptrs.iter().enumerate() {
        let (access, value) = read_pointer_register(&mut replay, pointer, pointer_max_bits)?;
        rs_accesses.push(access);
        rs_vals[i] = value;
    }
    let (rd_access, rd_val) = read_pointer_register(&mut replay, rd_ptr, pointer_max_bits)?;

    let mut reads = [[[0u16; BLOCK_FE_WIDTH]; INT256_NUM_MEMORY_BLOCKS]; NUM_READS];
    let mut read_accesses = Vec::with_capacity(NUM_READS * INT256_NUM_MEMORY_BLOCKS);
    for i in 0..NUM_READS {
        for (j, block) in reads[i].iter_mut().enumerate() {
            let byte_pointer = rs_vals[i] + (j * MEMORY_BLOCK_BYTES) as u32;
            let access = replay.read_u16(RV64_MEMORY_AS, byte_pointer / 2)?;
            *block = access.value;
            read_accesses.push(access);
        }
    }
    let inputs = reads.map(flatten_u16_blocks);
    let output = compute(inputs);
    let output_blocks = split_u16_blocks(output);
    let mut write_accesses = Vec::with_capacity(INT256_NUM_MEMORY_BLOCKS);
    for (j, block) in output_blocks.into_iter().enumerate() {
        let byte_pointer = rd_val + (j * MEMORY_BLOCK_BYTES) as u32;
        write_accesses.push(replay.write_u16(RV64_MEMORY_AS, byte_pointer / 2, block)?);
    }
    replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

    for pointer in rs_vals.into_iter().chain(once(rd_val)) {
        range_checker.add_count(ptr_bound_from_ptr(pointer, pointer_max_bits), U16_BITS);
    }
    for (access, cols) in write_accesses.iter().zip(&mut adapter_row.writes_aux) {
        cols.set_prev_data(access.previous_value.map(F::from_u16));
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    for (access, cols) in read_accesses.iter().zip(
        adapter_row
            .reads_aux
            .iter_mut()
            .flat_map(|blocks| blocks.iter_mut()),
    ) {
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    mem_helper.fill(
        rd_access.previous_timestamp,
        rd_access.timestamp,
        adapter_row.rd_read_aux.as_mut(),
    );
    for (access, cols) in rs_accesses.iter().zip(&mut adapter_row.rs_read_aux) {
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    adapter_row.rd_val = ptr_to_u16_limbs(rd_val).map(F::from_u16);
    adapter_row.rs_val = rs_vals.map(|pointer| ptr_to_u16_limbs(pointer).map(F::from_u16));
    adapter_row.rd_ptr = F::from_u32(rd_ptr);
    adapter_row.rs_ptr = rs_ptrs.map(F::from_u32);
    adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
    adapter_row.from_state.pc = F::from_u32(from_pc);

    Ok(AluReplay { inputs, output })
}

fn replay_alu_bytes<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
    mem_helper: &MemoryAuxColsFactory<F>,
    range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    adapter_row: &mut AluByteCols<F>,
    compute: impl FnOnce([[u8; INT256_NUM_U8_LIMBS]; NUM_READS]) -> [u8; INT256_NUM_U8_LIMBS],
) -> Result<AluReplay<u8, INT256_NUM_U8_LIMBS>, PostflightError> {
    let instruction = postflight.instruction(step);
    validate_alu_instruction(instruction)?;
    let rs_ptrs = [
        instruction.b.as_canonical_u32(),
        instruction.c.as_canonical_u32(),
    ];
    let rd_ptr = instruction.a.as_canonical_u32();
    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let mut replay = postflight.replay(step);

    let mut rs_accesses = Vec::with_capacity(NUM_READS);
    let mut rs_vals = [0u32; NUM_READS];
    for (i, &pointer) in rs_ptrs.iter().enumerate() {
        let (access, value) = read_pointer_register(&mut replay, pointer, pointer_max_bits)?;
        rs_accesses.push(access);
        rs_vals[i] = value;
    }
    let (rd_access, rd_val) = read_pointer_register(&mut replay, rd_ptr, pointer_max_bits)?;

    let mut reads = [[[0u8; MEMORY_BLOCK_BYTES]; INT256_NUM_MEMORY_BLOCKS]; NUM_READS];
    let mut read_accesses = Vec::with_capacity(NUM_READS * INT256_NUM_MEMORY_BLOCKS);
    for i in 0..NUM_READS {
        for (j, block) in reads[i].iter_mut().enumerate() {
            let byte_pointer = rs_vals[i] + (j * MEMORY_BLOCK_BYTES) as u32;
            let access = replay.read_u16(RV64_MEMORY_AS, byte_pointer / 2)?;
            *block = u16_block_to_bytes(access.value);
            read_accesses.push(access);
        }
    }
    let inputs = reads.map(flatten_byte_blocks);
    let output = compute(inputs);
    let output_blocks = split_byte_blocks(output);
    let mut write_accesses = Vec::with_capacity(INT256_NUM_MEMORY_BLOCKS);
    for (j, block) in output_blocks.into_iter().enumerate() {
        let byte_pointer = rd_val + (j * MEMORY_BLOCK_BYTES) as u32;
        write_accesses.push(replay.write_u16(
            RV64_MEMORY_AS,
            byte_pointer / 2,
            bytes_to_u16_block(block),
        )?);
    }
    replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

    for pointer in rs_vals.into_iter().chain(once(rd_val)) {
        range_checker.add_count(ptr_bound_from_ptr(pointer, pointer_max_bits), U16_BITS);
    }
    for (access, cols) in write_accesses.iter().zip(&mut adapter_row.writes_aux) {
        cols.set_prev_data(access.previous_value.map(F::from_u16));
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    for (access, cols) in read_accesses.iter().zip(
        adapter_row
            .reads_aux
            .iter_mut()
            .flat_map(|blocks| blocks.iter_mut()),
    ) {
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    mem_helper.fill(
        rd_access.previous_timestamp,
        rd_access.timestamp,
        adapter_row.rd_read_aux.as_mut(),
    );
    for (access, cols) in rs_accesses.iter().zip(&mut adapter_row.rs_read_aux) {
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    adapter_row.rd_val = ptr_to_u16_limbs(rd_val).map(F::from_u16);
    adapter_row.rs_val = rs_vals.map(|pointer| ptr_to_u16_limbs(pointer).map(F::from_u16));
    adapter_row.rd_ptr = F::from_u32(rd_ptr);
    adapter_row.rs_ptr = rs_ptrs.map(F::from_u32);
    adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
    adapter_row.from_state.pc = F::from_u32(from_pc);

    Ok(AluReplay { inputs, output })
}

fn replay_branch<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
    mem_helper: &MemoryAuxColsFactory<F>,
    range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    adapter_row: &mut BranchCols<F>,
    branch: impl FnOnce([[u16; INT256_NUM_U16_LIMBS]; NUM_READS]) -> bool,
) -> Result<([[u16; INT256_NUM_U16_LIMBS]; NUM_READS], bool), PostflightError> {
    let instruction = postflight.instruction(step);
    validate_alu_instruction(instruction)?;
    let rs_ptrs = [
        instruction.a.as_canonical_u32(),
        instruction.b.as_canonical_u32(),
    ];
    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let mut replay = postflight.replay(step);

    let mut rs_accesses = Vec::with_capacity(NUM_READS);
    let mut rs_vals = [0u32; NUM_READS];
    for (i, &pointer) in rs_ptrs.iter().enumerate() {
        let (access, value) = read_pointer_register(&mut replay, pointer, pointer_max_bits)?;
        rs_accesses.push(access);
        rs_vals[i] = value;
    }
    let mut reads = [[[0u16; BLOCK_FE_WIDTH]; INT256_NUM_MEMORY_BLOCKS]; NUM_READS];
    let mut read_accesses = Vec::with_capacity(NUM_READS * INT256_NUM_MEMORY_BLOCKS);
    for i in 0..NUM_READS {
        for (j, block) in reads[i].iter_mut().enumerate() {
            let byte_pointer = rs_vals[i] + (j * MEMORY_BLOCK_BYTES) as u32;
            let access = replay.read_u16(RV64_MEMORY_AS, byte_pointer / 2)?;
            *block = access.value;
            read_accesses.push(access);
        }
    }
    let inputs = reads.map(flatten_u16_blocks);
    let taken = branch(inputs);
    let next_pc = if taken {
        (F::from_u32(from_pc) + instruction.c).as_canonical_u32()
    } else {
        from_pc.wrapping_add(DEFAULT_PC_STEP)
    };
    replay.finish(next_pc)?;

    for pointer in rs_vals {
        range_checker.add_count(ptr_bound_from_ptr(pointer, pointer_max_bits), U16_BITS);
    }
    for (access, cols) in read_accesses.iter().zip(
        adapter_row
            .reads_aux
            .iter_mut()
            .flat_map(|blocks| blocks.iter_mut()),
    ) {
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    for (access, cols) in rs_accesses.iter().zip(&mut adapter_row.rs_read_aux) {
        mem_helper.fill(access.previous_timestamp, access.timestamp, cols.as_mut());
    }
    adapter_row.rs_val = rs_vals.map(|pointer| ptr_to_u16_limbs(pointer).map(F::from_u16));
    adapter_row.rs_ptr = rs_ptrs.map(F::from_u32);
    adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
    adapter_row.from_state.pc = F::from_u32(from_pc);
    Ok((inputs, taken))
}

fn opcodes_rows<F: PrimeField32>(postflight: &Postflight<'_, F>, opcodes: &[VmOpcode]) -> usize {
    opcodes
        .iter()
        .map(|&opcode| postflight.steps(opcode).len())
        .sum()
}

fn trace<F: PrimeField32>(rows: usize, width: usize) -> RowMajorMatrix<F> {
    RowMajorMatrix::new(F::zero_vec(next_power_of_two_or_zero(rows) * width), width)
}

fn add_sub(
    opcode: BaseAluOpcode,
    [b, c]: [[u16; INT256_NUM_U16_LIMBS]; NUM_READS],
) -> [u16; INT256_NUM_U16_LIMBS] {
    let mut output = [0u16; INT256_NUM_U16_LIMBS];
    let mut carry = 0u32;
    for i in 0..INT256_NUM_U16_LIMBS {
        output[i] = match opcode {
            BaseAluOpcode::ADD => {
                let value = u32::from(b[i]) + u32::from(c[i]) + carry;
                carry = value >> U16_BITS;
                value as u16
            }
            BaseAluOpcode::SUB => {
                let rhs = u32::from(c[i]) + carry;
                if u32::from(b[i]) >= rhs {
                    carry = 0;
                    (u32::from(b[i]) - rhs) as u16
                } else {
                    carry = 1;
                    (u32::from(b[i]) + (1 << U16_BITS) - rhs) as u16
                }
            }
            _ => unreachable!(),
        };
    }
    output
}

pub(crate) fn generate_add_sub_trace<F: PrimeField32>(
    chip: &Rv64AddSub256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [BaseAluOpcode::ADD, BaseAluOpcode::SUB];
    let global = opcodes.map(|opcode| Rv64BaseAlu256Opcode(opcode).global_opcode());
    let adapter_width = AluU16Cols::<F>::width();
    let width = adapter_width + AddSubCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width();
    let mut trace = trace(opcodes_rows(postflight, &global), width);
    let mut row_index = 0;
    for (opcode, global_opcode) in opcodes.into_iter().zip(global) {
        let steps = postflight.steps(global_opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let replay = replay_alu_u16(
                postflight,
                step,
                pointer_max_bits,
                &chip.mem_helper.as_borrowed(),
                &chip.inner.range_checker_chip,
                adapter_row.borrow_mut(),
                |inputs| add_sub(opcode, inputs),
            )?;
            let core: &mut AddSubCoreCols<F, INT256_NUM_U16_LIMBS, U16_BITS> =
                core_row.borrow_mut();
            core.opcode_add_flag = F::from_bool(opcode == BaseAluOpcode::ADD);
            core.opcode_sub_flag = F::from_bool(opcode == BaseAluOpcode::SUB);
            for &value in &replay.output {
                chip.inner
                    .range_checker_chip
                    .add_count(value as u32, U16_BITS);
            }
            core.a = replay.output.map(F::from_u16);
            core.b = replay.inputs[0].map(F::from_u16);
            core.c = replay.inputs[1].map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }
    Ok(trace)
}

pub(crate) fn generate_bitwise_trace<F: PrimeField32>(
    chip: &Rv64BitwiseLogic256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
    range_checker: &openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND];
    let global = opcodes.map(|opcode| Rv64BaseAlu256Opcode(opcode).global_opcode());
    let adapter_width = AluByteCols::<F>::width();
    let width =
        adapter_width + BitwiseLogicCoreCols::<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>::width();
    let mut trace = trace(opcodes_rows(postflight, &global), width);
    let mut row_index = 0;
    for (opcode, global_opcode) in opcodes.into_iter().zip(global) {
        let steps = postflight.steps(global_opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let replay = replay_alu_bytes(
                postflight,
                step,
                pointer_max_bits,
                &chip.mem_helper.as_borrowed(),
                range_checker,
                adapter_row.borrow_mut(),
                |[b, c]| {
                    array::from_fn(|i| match opcode {
                        BaseAluOpcode::XOR => b[i] ^ c[i],
                        BaseAluOpcode::OR => b[i] | c[i],
                        BaseAluOpcode::AND => b[i] & c[i],
                        _ => unreachable!(),
                    })
                },
            )?;
            let core: &mut BitwiseLogicCoreCols<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS> =
                core_row.borrow_mut();
            core.opcode_xor_flag = F::from_bool(opcode == BaseAluOpcode::XOR);
            core.opcode_or_flag = F::from_bool(opcode == BaseAluOpcode::OR);
            core.opcode_and_flag = F::from_bool(opcode == BaseAluOpcode::AND);
            for (&b, &c) in replay.inputs[0].iter().zip(&replay.inputs[1]) {
                chip.inner
                    .bitwise_lookup_chip
                    .request_xor(b as u32, c as u32);
            }
            core.a = replay.output.map(F::from_u8);
            core.b = replay.inputs[0].map(F::from_u8);
            core.c = replay.inputs[1].map(F::from_u8);
            Ok(())
        })?;
        row_index += steps.len();
    }
    Ok(trace)
}

fn less_than(
    signed: bool,
    b: &[u16; INT256_NUM_U16_LIMBS],
    c: &[u16; INT256_NUM_U16_LIMBS],
) -> (bool, usize, bool, bool) {
    let b_sign = signed && b[INT256_NUM_U16_LIMBS - 1] >> (U16_BITS - 1) == 1;
    let c_sign = signed && c[INT256_NUM_U16_LIMBS - 1] >> (U16_BITS - 1) == 1;
    for i in (0..INT256_NUM_U16_LIMBS).rev() {
        if b[i] != c[i] {
            return ((b[i] < c[i]) ^ b_sign ^ c_sign, i, b_sign, c_sign);
        }
    }
    (false, INT256_NUM_U16_LIMBS, b_sign, c_sign)
}

fn fill_less_than<F: PrimeField32>(
    chip: &Rv64LessThan256Chip<F>,
    core: &mut LessThanCoreCols<F, INT256_NUM_U16_LIMBS, U16_BITS>,
    opcode: LessThanOpcode,
    [b, c]: [[u16; INT256_NUM_U16_LIMBS]; NUM_READS],
) {
    let signed = opcode == LessThanOpcode::SLT;
    let (cmp_result, diff_idx, b_sign, c_sign) = less_than(signed, &b, &c);
    let (b_msb_f, b_msb_range) = if b_sign {
        (
            -F::from_u16(b[INT256_NUM_U16_LIMBS - 1].wrapping_neg()),
            b[INT256_NUM_U16_LIMBS - 1] as u32 - (1 << (U16_BITS - 1)),
        )
    } else {
        (
            F::from_u16(b[INT256_NUM_U16_LIMBS - 1]),
            b[INT256_NUM_U16_LIMBS - 1] as u32 + ((signed as u32) << (U16_BITS - 1)),
        )
    };
    let (c_msb_f, c_msb_range) = if c_sign {
        (
            -F::from_u16(c[INT256_NUM_U16_LIMBS - 1].wrapping_neg()),
            c[INT256_NUM_U16_LIMBS - 1] as u32 - (1 << (U16_BITS - 1)),
        )
    } else {
        (
            F::from_u16(c[INT256_NUM_U16_LIMBS - 1]),
            c[INT256_NUM_U16_LIMBS - 1] as u32 + ((signed as u32) << (U16_BITS - 1)),
        )
    };
    core.diff_val = if diff_idx == INT256_NUM_U16_LIMBS {
        F::ZERO
    } else if diff_idx == INT256_NUM_U16_LIMBS - 1 {
        if cmp_result {
            c_msb_f - b_msb_f
        } else {
            b_msb_f - c_msb_f
        }
    } else if cmp_result {
        F::from_u16((c[diff_idx] as u32 - b[diff_idx] as u32) as u16)
    } else {
        F::from_u16((b[diff_idx] as u32 - c[diff_idx] as u32) as u16)
    };
    chip.inner
        .range_checker_chip
        .add_count(b_msb_range, U16_BITS);
    chip.inner
        .range_checker_chip
        .add_count(c_msb_range, U16_BITS);
    core.diff_marker = [F::ZERO; INT256_NUM_U16_LIMBS];
    if diff_idx != INT256_NUM_U16_LIMBS {
        chip.inner
            .range_checker_chip
            .add_count(core.diff_val.as_canonical_u32() - 1, U16_BITS);
        core.diff_marker[diff_idx] = F::ONE;
    }
    core.b_msb_f = b_msb_f;
    core.c_msb_f = c_msb_f;
    core.opcode_slt_flag = F::from_bool(signed);
    core.opcode_sltu_flag = F::from_bool(!signed);
    core.cmp_result = F::from_bool(cmp_result);
    core.b = b.map(F::from_u16);
    core.c = c.map(F::from_u16);
}

pub(crate) fn generate_less_than_trace<F: PrimeField32>(
    chip: &Rv64LessThan256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [LessThanOpcode::SLT, LessThanOpcode::SLTU];
    let global = opcodes
        .iter()
        .map(|&opcode| Rv64LessThan256Opcode(opcode).global_opcode())
        .collect::<Vec<_>>();
    let adapter_width = AluU16Cols::<F>::width();
    let width = adapter_width + LessThanCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width();
    let mut trace = trace(opcodes_rows(postflight, &global), width);
    let mut row_index = 0;
    for (opcode, global_opcode) in opcodes.into_iter().zip(global) {
        let steps = postflight.steps(global_opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let replay = replay_alu_u16(
                postflight,
                step,
                pointer_max_bits,
                &chip.mem_helper.as_borrowed(),
                &chip.inner.range_checker_chip,
                adapter_row.borrow_mut(),
                |inputs| {
                    let mut output = [0u16; INT256_NUM_U16_LIMBS];
                    output[0] =
                        less_than(opcode == LessThanOpcode::SLT, &inputs[0], &inputs[1]).0 as u16;
                    output
                },
            )?;
            fill_less_than(chip, core_row.borrow_mut(), opcode, replay.inputs);
            Ok(())
        })?;
        row_index += steps.len();
    }
    Ok(trace)
}

fn branch_eq(opcode: BranchEqualOpcode, [a, b]: [[u16; INT256_NUM_U16_LIMBS]; NUM_READS]) -> bool {
    match opcode {
        BranchEqualOpcode::BEQ => a == b,
        BranchEqualOpcode::BNE => a != b,
    }
}

pub(crate) fn generate_branch_equal_trace<F: PrimeField32>(
    chip: &Rv64BranchEqual256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
    range_checker: &openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [BranchEqualOpcode::BEQ, BranchEqualOpcode::BNE];
    let global = opcodes
        .iter()
        .map(|&opcode| Rv64BranchEqual256Opcode(opcode).global_opcode())
        .collect::<Vec<_>>();
    let adapter_width = BranchCols::<F>::width();
    let width = adapter_width + BranchEqualCoreCols::<F, INT256_NUM_U16_LIMBS>::width();
    let mut trace = trace(opcodes_rows(postflight, &global), width);
    let mut row_index = 0;
    for (opcode, global_opcode) in opcodes.into_iter().zip(global) {
        let steps = postflight.steps(global_opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let (inputs, cmp_result) = replay_branch(
                postflight,
                step,
                pointer_max_bits,
                &chip.mem_helper.as_borrowed(),
                range_checker,
                adapter_row.borrow_mut(),
                |inputs| branch_eq(opcode, inputs),
            )?;
            let [a, b] = inputs;
            let core: &mut BranchEqualCoreCols<F, INT256_NUM_U16_LIMBS> = core_row.borrow_mut();
            core.diff_inv_marker = [F::ZERO; INT256_NUM_U16_LIMBS];
            if let Some(index) = (0..INT256_NUM_U16_LIMBS).find(|&i| a[i] != b[i]) {
                core.diff_inv_marker[index] =
                    (F::from_u16(a[index]) - F::from_u16(b[index])).inverse();
            }
            core.opcode_beq_flag = F::from_bool(opcode == BranchEqualOpcode::BEQ);
            core.opcode_bne_flag = F::from_bool(opcode == BranchEqualOpcode::BNE);
            core.imm = postflight.instruction(step).c;
            core.cmp_result = F::from_bool(cmp_result);
            core.a = a.map(F::from_u16);
            core.b = b.map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }
    Ok(trace)
}

fn branch_compare(
    opcode: BranchLessThanOpcode,
    a: &[u16; INT256_NUM_U16_LIMBS],
    b: &[u16; INT256_NUM_U16_LIMBS],
) -> (bool, usize, bool, bool) {
    let signed = matches!(
        opcode,
        BranchLessThanOpcode::BLT | BranchLessThanOpcode::BGE
    );
    let ge = matches!(
        opcode,
        BranchLessThanOpcode::BGE | BranchLessThanOpcode::BGEU
    );
    let a_sign = signed && a[INT256_NUM_U16_LIMBS - 1] >> (U16_BITS - 1) == 1;
    let b_sign = signed && b[INT256_NUM_U16_LIMBS - 1] >> (U16_BITS - 1) == 1;
    for i in (0..INT256_NUM_U16_LIMBS).rev() {
        if a[i] != b[i] {
            return ((a[i] < b[i]) ^ a_sign ^ b_sign ^ ge, i, a_sign, b_sign);
        }
    }
    (ge, INT256_NUM_U16_LIMBS, a_sign, b_sign)
}

pub(crate) fn generate_branch_less_than_trace<F: PrimeField32>(
    chip: &Rv64BranchLessThan256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [
        BranchLessThanOpcode::BLT,
        BranchLessThanOpcode::BLTU,
        BranchLessThanOpcode::BGE,
        BranchLessThanOpcode::BGEU,
    ];
    let global = opcodes
        .iter()
        .map(|&opcode| Rv64BranchLessThan256Opcode(opcode).global_opcode())
        .collect::<Vec<_>>();
    let adapter_width = BranchCols::<F>::width();
    let width =
        adapter_width + BranchLessThanCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width();
    let mut trace = trace(opcodes_rows(postflight, &global), width);
    let mut row_index = 0;
    for (opcode, global_opcode) in opcodes.into_iter().zip(global) {
        let steps = postflight.steps(global_opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let (inputs, cmp_result) = replay_branch(
                postflight,
                step,
                pointer_max_bits,
                &chip.mem_helper.as_borrowed(),
                &chip.inner.range_checker_chip,
                adapter_row.borrow_mut(),
                |[a, b]| branch_compare(opcode, &a, &b).0,
            )?;
            let [a, b] = inputs;
            let (_, diff_idx, a_sign, b_sign) = branch_compare(opcode, &a, &b);
            let signed = matches!(
                opcode,
                BranchLessThanOpcode::BLT | BranchLessThanOpcode::BGE
            );
            let ge = matches!(
                opcode,
                BranchLessThanOpcode::BGE | BranchLessThanOpcode::BGEU
            );
            let cmp_lt = cmp_result ^ ge;
            let (a_msb_f, a_msb_range) =
                signed_msb::<F>(a[INT256_NUM_U16_LIMBS - 1], signed, a_sign);
            let (b_msb_f, b_msb_range) =
                signed_msb::<F>(b[INT256_NUM_U16_LIMBS - 1], signed, b_sign);
            let core: &mut BranchLessThanCoreCols<F, INT256_NUM_U16_LIMBS, U16_BITS> =
                core_row.borrow_mut();
            core.diff_val = comparison_diff(a, b, diff_idx, cmp_lt, a_msb_f, b_msb_f);
            chip.inner
                .range_checker_chip
                .add_count(a_msb_range, U16_BITS);
            chip.inner
                .range_checker_chip
                .add_count(b_msb_range, U16_BITS);
            core.diff_marker = [F::ZERO; INT256_NUM_U16_LIMBS];
            if diff_idx != INT256_NUM_U16_LIMBS {
                chip.inner
                    .range_checker_chip
                    .add_count(core.diff_val.as_canonical_u32() - 1, U16_BITS);
                core.diff_marker[diff_idx] = F::ONE;
            }
            core.cmp_lt = F::from_bool(cmp_lt);
            core.a_msb_f = a_msb_f;
            core.b_msb_f = b_msb_f;
            core.opcode_blt_flag = F::from_bool(opcode == BranchLessThanOpcode::BLT);
            core.opcode_bltu_flag = F::from_bool(opcode == BranchLessThanOpcode::BLTU);
            core.opcode_bge_flag = F::from_bool(opcode == BranchLessThanOpcode::BGE);
            core.opcode_bgeu_flag = F::from_bool(opcode == BranchLessThanOpcode::BGEU);
            core.imm = postflight.instruction(step).c;
            core.cmp_result = F::from_bool(cmp_result);
            core.a = a.map(F::from_u16);
            core.b = b.map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }
    Ok(trace)
}

fn signed_msb<F: PrimeField32>(limb: u16, signed: bool, sign: bool) -> (F, u32) {
    if sign {
        (
            -F::from_u16(limb.wrapping_neg()),
            limb as u32 - (1 << (U16_BITS - 1)),
        )
    } else {
        (
            F::from_u16(limb),
            limb as u32 + ((signed as u32) << (U16_BITS - 1)),
        )
    }
}

fn comparison_diff<F: PrimeField32>(
    a: [u16; INT256_NUM_U16_LIMBS],
    b: [u16; INT256_NUM_U16_LIMBS],
    diff_idx: usize,
    cmp_lt: bool,
    a_msb_f: F,
    b_msb_f: F,
) -> F {
    if diff_idx == INT256_NUM_U16_LIMBS {
        F::ZERO
    } else if diff_idx == INT256_NUM_U16_LIMBS - 1 {
        if cmp_lt {
            b_msb_f - a_msb_f
        } else {
            a_msb_f - b_msb_f
        }
    } else if cmp_lt {
        F::from_u16((b[diff_idx] as u32 - a[diff_idx] as u32) as u16)
    } else {
        F::from_u16((a[diff_idx] as u32 - b[diff_idx] as u32) as u16)
    }
}

fn mul_with_carry(
    x: &[u8; INT256_NUM_U8_LIMBS],
    y: &[u8; INT256_NUM_U8_LIMBS],
) -> ([u8; INT256_NUM_U8_LIMBS], [u32; INT256_NUM_U8_LIMBS]) {
    let mut result = [0u8; INT256_NUM_U8_LIMBS];
    let mut carry = [0u32; INT256_NUM_U8_LIMBS];
    for i in 0..INT256_NUM_U8_LIMBS {
        let mut value = if i == 0 { 0 } else { carry[i - 1] };
        for j in 0..=i {
            value += u32::from(x[j]) * u32::from(y[i - j]);
        }
        carry[i] = value >> RV64_BYTE_BITS;
        result[i] = value as u8;
    }
    (result, carry)
}

pub(crate) fn generate_multiplication_trace<F: PrimeField32>(
    chip: &Rv64Multiplication256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
    range_checker: &openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = Rv64Mul256Opcode(MulOpcode::MUL).global_opcode();
    let adapter_width = AluByteCols::<F>::width();
    let width =
        adapter_width + MultiplicationCoreCols::<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>::width();
    let mut trace = trace(postflight.steps(opcode).len(), width);
    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let replay = replay_alu_bytes(
            postflight,
            step,
            pointer_max_bits,
            &chip.mem_helper.as_borrowed(),
            range_checker,
            adapter_row.borrow_mut(),
            |[b, c]| u256_mul(b, c),
        )?;
        let (output, carry) = mul_with_carry(&replay.inputs[0], &replay.inputs[1]);
        if output != replay.output {
            return Err(invalid(
                "int256 multiplication replay produced inconsistent output",
            ));
        }
        for (&a, &carry) in output.iter().zip(&carry) {
            chip.inner
                .range_tuple_chip
                .add_count(&[u32::from(a), carry]);
        }
        for (&b, &c) in replay.inputs[0].iter().zip(&replay.inputs[1]) {
            chip.inner
                .bitwise_lookup_chip
                .request_range(u32::from(b), u32::from(c));
        }
        let core: &mut MultiplicationCoreCols<F, INT256_NUM_U8_LIMBS, RV64_BYTE_BITS> =
            core_row.borrow_mut();
        core.is_valid = F::ONE;
        core.a = output.map(F::from_u8);
        core.b = replay.inputs[0].map(F::from_u8);
        core.c = replay.inputs[1].map(F::from_u8);
        Ok(())
    })?;
    Ok(trace)
}

fn shift_amount(c: &[u16; INT256_NUM_U16_LIMBS]) -> (usize, usize) {
    let shift = usize::from(c[0]) % (INT256_NUM_U16_LIMBS * U16_BITS);
    (shift / U16_BITS, shift % U16_BITS)
}

fn shift_logical(
    opcode: ShiftOpcode,
    b: &[u16; INT256_NUM_U16_LIMBS],
    c: &[u16; INT256_NUM_U16_LIMBS],
) -> ([u16; INT256_NUM_U16_LIMBS], usize, usize) {
    let (limb_shift, bit_shift) = shift_amount(c);
    let mut output = [0u16; INT256_NUM_U16_LIMBS];
    match opcode {
        ShiftOpcode::SLL => {
            for i in limb_shift..INT256_NUM_U16_LIMBS {
                let mut value = u32::from(b[i - limb_shift]) << bit_shift;
                if i > limb_shift && bit_shift > 0 {
                    value |= u32::from(b[i - limb_shift - 1]) >> (U16_BITS - bit_shift);
                }
                output[i] = value as u16;
            }
        }
        ShiftOpcode::SRL => {
            for i in 0..INT256_NUM_U16_LIMBS - limb_shift {
                let mut value = u32::from(b[i + limb_shift]) >> bit_shift;
                if i + limb_shift + 1 < INT256_NUM_U16_LIMBS && bit_shift > 0 {
                    value |= u32::from(b[i + limb_shift + 1]) << (U16_BITS - bit_shift);
                }
                output[i] = value as u16;
            }
        }
        _ => unreachable!("logical shift generator received non-logical opcode"),
    }
    (output, limb_shift, bit_shift)
}

fn fill_shift_decomposition<F: PrimeField32>(
    range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    b: &[u16; INT256_NUM_U16_LIMBS],
    c: &[u16; INT256_NUM_U16_LIMBS],
    limb_shift: usize,
    bit_shift: usize,
    left: bool,
) -> (
    [F; INT256_NUM_U16_LIMBS],
    [F; INT256_NUM_U16_LIMBS],
    [F; INT256_NUM_U16_LIMBS],
    [F; U16_BITS],
) {
    let num_bits_log = (INT256_NUM_U16_LIMBS * U16_BITS).ilog2() as usize;
    range_checker.add_count(
        ((usize::from(c[0]) - bit_shift - limb_shift * U16_BITS) >> num_bits_log) as u32,
        U16_BITS - num_bits_log,
    );
    let aux_bits = U16_BITS - bit_shift;
    let mut carry = [F::ZERO; INT256_NUM_U16_LIMBS];
    let mut aux = [F::ZERO; INT256_NUM_U16_LIMBS];
    for i in 0..INT256_NUM_U16_LIMBS {
        let limb = u32::from(b[i]);
        let (carry_value, aux_value) = if left {
            (limb >> aux_bits, limb & ((1u32 << aux_bits) - 1))
        } else {
            (limb & ((1u32 << bit_shift) - 1), limb >> bit_shift)
        };
        range_checker.add_count(carry_value, bit_shift);
        range_checker.add_count(aux_value, aux_bits);
        carry[i] = F::from_u32(carry_value);
        aux[i] = F::from_u32(aux_value);
    }
    let mut limb_marker = [F::ZERO; INT256_NUM_U16_LIMBS];
    limb_marker[limb_shift] = F::ONE;
    let mut bit_marker = [F::ZERO; U16_BITS];
    bit_marker[bit_shift] = F::ONE;
    (carry, aux, limb_marker, bit_marker)
}

pub(crate) fn generate_shift_logical_trace<F: PrimeField32>(
    chip: &Rv64ShiftLogical256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [ShiftOpcode::SLL, ShiftOpcode::SRL];
    let global = opcodes.map(|opcode| Rv64Shift256Opcode(opcode).global_opcode());
    let adapter_width = AluU16Cols::<F>::width();
    let width = adapter_width + ShiftLogicalCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width();
    let mut trace = trace(opcodes_rows(postflight, &global), width);
    let mut row_index = 0;
    for (opcode, global_opcode) in opcodes.into_iter().zip(global) {
        let steps = postflight.steps(global_opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let mut shift = None;
            let replay = replay_alu_u16(
                postflight,
                step,
                pointer_max_bits,
                &chip.mem_helper.as_borrowed(),
                &chip.inner.range_checker_chip,
                adapter_row.borrow_mut(),
                |[b, c]| {
                    let result = shift_logical(opcode, &b, &c);
                    shift = Some((result.1, result.2));
                    result.0
                },
            )?;
            let (limb_shift, bit_shift) =
                shift.expect("logical-shift replay called its compute closure");
            let (carry, aux, limb_marker, bit_marker) = fill_shift_decomposition(
                &chip.inner.range_checker_chip,
                &replay.inputs[0],
                &replay.inputs[1],
                limb_shift,
                bit_shift,
                opcode == ShiftOpcode::SLL,
            );
            let core: &mut ShiftLogicalCoreCols<F, INT256_NUM_U16_LIMBS, U16_BITS> =
                core_row.borrow_mut();
            let aux_bits = U16_BITS - bit_shift;
            core.carry_multiplier_left = if opcode == ShiftOpcode::SLL {
                F::from_u32(1 << aux_bits)
            } else {
                F::ZERO
            };
            core.bit_multiplier_left = if opcode == ShiftOpcode::SLL {
                F::from_u32(1 << bit_shift)
            } else {
                F::ZERO
            };
            core.opcode_sll_flag = F::from_bool(opcode == ShiftOpcode::SLL);
            core.bit_shift_carry = carry;
            core.bit_shift_aux = aux;
            core.limb_shift_marker = limb_marker;
            core.bit_shift_marker = bit_marker;
            core.a = replay.output.map(F::from_u16);
            core.b = replay.inputs[0].map(F::from_u16);
            core.c = replay.inputs[1].map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }
    Ok(trace)
}

fn shift_arithmetic(
    b: &[u16; INT256_NUM_U16_LIMBS],
    c: &[u16; INT256_NUM_U16_LIMBS],
) -> ([u16; INT256_NUM_U16_LIMBS], usize, usize) {
    let fill = u16::MAX * (b[INT256_NUM_U16_LIMBS - 1] >> (U16_BITS - 1));
    let (limb_shift, bit_shift) = shift_amount(c);
    let mut output = [fill; INT256_NUM_U16_LIMBS];
    for i in 0..INT256_NUM_U16_LIMBS - limb_shift {
        let mut value = u32::from(b[i + limb_shift]) >> bit_shift;
        if bit_shift > 0 {
            let upper = if i + limb_shift + 1 < INT256_NUM_U16_LIMBS {
                b[i + limb_shift + 1]
            } else {
                fill
            };
            value |= u32::from(upper) << (U16_BITS - bit_shift);
        }
        output[i] = value as u16;
    }
    (output, limb_shift, bit_shift)
}

pub(crate) fn generate_shift_arithmetic_trace<F: PrimeField32>(
    chip: &Rv64ShiftRightArithmetic256Chip<F>,
    postflight: &Postflight<'_, F>,
    pointer_max_bits: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = Rv64Shift256Opcode(ShiftOpcode::SRA).global_opcode();
    let adapter_width = AluU16Cols::<F>::width();
    let width =
        adapter_width + ShiftRightArithmeticCoreCols::<F, INT256_NUM_U16_LIMBS, U16_BITS>::width();
    let mut trace = trace(postflight.steps(opcode).len(), width);
    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let mut shift = None;
        let replay = replay_alu_u16(
            postflight,
            step,
            pointer_max_bits,
            &chip.mem_helper.as_borrowed(),
            &chip.inner.range_checker_chip,
            adapter_row.borrow_mut(),
            |[b, c]| {
                let result = shift_arithmetic(&b, &c);
                shift = Some((result.1, result.2));
                result.0
            },
        )?;
        let (limb_shift, bit_shift) =
            shift.expect("arithmetic-shift replay called its compute closure");
        let (carry, aux, limb_marker, bit_marker) = fill_shift_decomposition(
            &chip.inner.range_checker_chip,
            &replay.inputs[0],
            &replay.inputs[1],
            limb_shift,
            bit_shift,
            false,
        );
        let b_sign = replay.inputs[0][INT256_NUM_U16_LIMBS - 1] >> (U16_BITS - 1);
        chip.inner.range_checker_chip.add_count(
            replay.inputs[0][INT256_NUM_U16_LIMBS - 1] as u32 - ((b_sign as u32) << (U16_BITS - 1)),
            U16_BITS - 1,
        );
        let core: &mut ShiftRightArithmeticCoreCols<F, INT256_NUM_U16_LIMBS, U16_BITS> =
            core_row.borrow_mut();
        core.b_sign = F::from_u16(b_sign);
        core.bit_shift_carry = carry;
        core.bit_shift_aux = aux;
        core.limb_shift_marker = limb_marker;
        core.bit_shift_marker = bit_marker;
        core.a = replay.output.map(F::from_u16);
        core.b = replay.inputs[0].map(F::from_u16);
        core.c = replay.inputs[1].map(F::from_u16);
        Ok(())
    })?;
    Ok(trace)
}
