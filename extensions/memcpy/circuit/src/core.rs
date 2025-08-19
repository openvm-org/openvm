use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    sync::Arc,
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{
            MemoryBaseAuxCols, MemoryBaseAuxRecord, MemoryBridge, MemoryReadAuxRecord,
            MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
        },
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    utils::{not, or, select},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::{bus::MemcpyBus, MemcpyIterChip};
use openvm_circuit::arch::{
    execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
    get_record_from_slice, ExecuteFunc, ExecutionError, Executor,
    MeteredExecutor, RecordArena, StaticProgramError, TraceFiller, VmExecState,
};
use openvm_memcpy_transpiler::Rv32MemcpyOpcode;

// Import constants from lib.rs
use crate::{
    A1_REGISTER_PTR, A2_REGISTER_PTR, A3_REGISTER_PTR, A4_REGISTER_PTR, MEMCPY_LOOP_LIMB_BITS,
    MEMCPY_LOOP_NUM_LIMBS,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct MemcpyLoopCols<T> {
    pub from_state: ExecutionState<T>,
    pub dest: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub source: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub len: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub shift: [T; 2],
    pub is_valid: T,
    pub to_timestamp: T,
    pub to_dest: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub to_source: [T; MEMCPY_LOOP_NUM_LIMBS],
    pub to_len: T,
    pub write_aux: [MemoryBaseAuxCols<T>; 3],
    pub source_minus_twelve_carry: T,
    pub to_source_minus_twelve_carry: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct MemcpyLoopAir {
    pub memory_bridge: MemoryBridge,
    pub execution_bridge: ExecutionBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub memcpy_bus: MemcpyBus,
    pub pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for MemcpyLoopAir {
    fn width(&self) -> usize {
        MemcpyLoopCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MemcpyLoopAir {}
impl<F: Field> PartitionedBaseAir<F> for MemcpyLoopAir {}

impl<AB: InteractionBuilder> Air<AB> for MemcpyLoopAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemcpyLoopCols<AB::Var> = (*local).borrow();

        let timestamp: AB::Var = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let from_le_bytes = |data: [AB::Var; 4]| {
            data.iter().fold(AB::Expr::ZERO, |acc, x| {
                acc * AB::Expr::from_canonical_u32(256) + *x
            })
        };

        let u8_word_to_u16 = |data: [AB::Var; 4]| {
            [
                data[0] + data[1] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS),
                data[2] + data[3] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS),
            ]
        };

        let shift = local.shift[1] * AB::Expr::from_canonical_u32(2) + local.shift[0];
        let is_shift_non_zero = or::<AB::Expr>(local.shift[0], local.shift[1]);
        let dest = from_le_bytes(local.dest);
        let source = from_le_bytes(local.source);
        let len = from_le_bytes(local.len);
        let to_dest = from_le_bytes(local.to_dest);
        let to_source = from_le_bytes(local.to_source);
        let to_len = local.to_len;

        builder.assert_bool(local.is_valid);
        for i in 0..2 {
            builder.assert_bool(local.shift[i]);
        }
        builder.assert_bool(local.source_minus_twelve_carry);
        builder.assert_bool(local.to_source_minus_twelve_carry);

        let mut shift_zero_when = builder.when(not::<AB::Expr>(is_shift_non_zero.clone()));
        shift_zero_when.assert_zero(local.source_minus_twelve_carry);
        shift_zero_when.assert_zero(local.to_source_minus_twelve_carry);

        // Write source and destination to registers
        let write_data = [
            (local.dest, local.to_dest, A1_REGISTER_PTR, A3_REGISTER_PTR),
            (
                local.source,
                local.to_source,
                A2_REGISTER_PTR,
                A4_REGISTER_PTR,
            ),
        ];

        write_data
            .iter()
            .enumerate()
            .for_each(|(idx, (dest, to_dest, ptr, zero_shift_ptr))| {
                let write_ptr = select::<AB::Expr>(
                    is_shift_non_zero.clone(),
                    AB::Expr::from_canonical_usize(*ptr),
                    AB::Expr::from_canonical_usize(*zero_shift_ptr),
                );

                self.memory_bridge
                    .write(
                        MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), write_ptr),
                        *to_dest,
                        timestamp_pp(),
                        &MemoryWriteAuxCols::from_base(local.write_aux[idx], *dest),
                    )
                    .eval(builder, local.is_valid);
            });

        // Write length to a2 register
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                    AB::Expr::from_canonical_usize(A2_REGISTER_PTR),
                ),
                [
                    to_len.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                timestamp_pp(),
                &MemoryWriteAuxCols::from_base(local.write_aux[2], local.len),
            )
            .eval(builder, local.is_valid);

        // Generate 16-bit limbs for range checking
        let len_u16_limbs = u8_word_to_u16(local.len);
        let dest_u16_limbs = u8_word_to_u16(local.dest);
        let to_dest_u16_limbs = u8_word_to_u16(local.to_dest);
        let source_u16_limbs = [
            local.source[0]
                + local.source[1] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - AB::Expr::from_canonical_u32(12) * is_shift_non_zero.clone()
                + local.source_minus_twelve_carry
                    * AB::F::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS)),
            local.source[2]
                + local.source[3] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - local.source_minus_twelve_carry,
        ];
        let to_source_u16_limbs = [
            local.to_source[0]
                + local.to_source[1] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - AB::Expr::from_canonical_u32(12) * is_shift_non_zero.clone()
                + local.to_source_minus_twelve_carry
                    * AB::F::from_canonical_u32(1 << (2 * MEMCPY_LOOP_LIMB_BITS)),
            local.to_source[2]
                + local.to_source[3] * AB::F::from_canonical_u32(1 << MEMCPY_LOOP_LIMB_BITS)
                - local.to_source_minus_twelve_carry,
        ];

        // Range check addresses and n
        let range_check_data = [
            (len_u16_limbs, false),
            (dest_u16_limbs, true),
            (source_u16_limbs, true),
            (to_dest_u16_limbs, true),
            (to_source_u16_limbs, true),
        ];

        range_check_data.iter().for_each(|(data, is_address)| {
            let (data_0, num_bits) = if *is_address {
                (
                    data[0].clone() * AB::F::from_canonical_u32(4).inverse(),
                    MEMCPY_LOOP_LIMB_BITS * 2 - 2,
                )
            } else {
                (data[0].clone(), MEMCPY_LOOP_LIMB_BITS * 2)
            };
            self.range_bus
                .range_check(data_0, num_bits)
                .eval(builder, local.is_valid);
            self.range_bus
                .range_check(
                    data[1].clone(),
                    self.pointer_max_bits - MEMCPY_LOOP_LIMB_BITS * 2,
                )
                .eval(builder, local.is_valid);
        });

        // Send message to memcpy call bus
        self.memcpy_bus
            .send(
                timestamp + AB::Expr::from_canonical_usize(timestamp_delta),
                dest - AB::Expr::from_canonical_u32(16),
                source
                    - select::<AB::Expr>(
                        is_shift_non_zero.clone(),
                        AB::Expr::from_canonical_u32(28),
                        AB::Expr::from_canonical_u32(16),
                    ),
                len.clone() - shift.clone(),
                shift.clone(),
            )
            .eval(builder, local.is_valid);

        // Receive message from memcpy return bus
        self.memcpy_bus
            .receive(
                local.to_timestamp,
                to_dest,
                to_source,
                to_len - shift.clone(),
                AB::Expr::from_canonical_u32(4),
            )
            .eval(builder, local.is_valid);

        // Make sure the request and response match
        builder.assert_eq(
            local.to_timestamp - (timestamp + AB::Expr::from_canonical_usize(timestamp_delta)),
            AB::Expr::TWO * (len.clone() - to_len) + is_shift_non_zero.clone(),
        );

        // Execution bus + program bus
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(Rv32MemcpyOpcode::MEMCPY_LOOP as usize),
                [shift.clone()],
                local.from_state,
                local.to_timestamp,
            )
            .eval(builder, local.is_valid);
    }
}

#[derive(derive_new::new, Clone, Copy)]
pub struct MemcpyLoopExecutor {}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct MemcpyLoopRecord {
    pub shift: [u8; 2],
    pub dest: [u8; MEMCPY_LOOP_NUM_LIMBS],
    pub source: [u8; MEMCPY_LOOP_NUM_LIMBS],
    pub len: [u8; MEMCPY_LOOP_NUM_LIMBS],
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub register_aux: [MemoryBaseAuxRecord; 3],
    pub memory_read_data: Vec<[u8; MEMCPY_LOOP_NUM_LIMBS]>,
    pub read_aux: Vec<MemoryReadAuxRecord>,
    pub write_aux: Vec<MemoryWriteBytesAuxRecord<4>>,
}

#[derive(derive_new::new)]
pub struct MemcpyLoopFiller {
    pub pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub memcpy_iter_chip: Arc<MemcpyIterChip>,
}

pub type MemcpyLoopChip<F> = VmChipWrapper<F, MemcpyLoopFiller>;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MemcpyLoopPreCompute {
    c: u8,
}

impl<F, RA> PreflightExecutor<F, RA> for MemcpyLoopExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, EmptyMultiRowLayout, &'buf mut MemcpyLoopRecord>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32MemcpyOpcode::MEMCPY_LOOP)
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, .. } = instruction;
        debug_assert_eq!(*opcode, Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode());
        let shift = c.as_canonical_u32() as u8;
        debug_assert!([0, 1, 2, 3].contains(&shift));
        let mut record = state.ctx.alloc(EmptyMultiRowLayout::default());

        let mut dest = read_rv32_register(
            state.memory.data(),
            if shift == 0 {
                A3_REGISTER_PTR
            } else {
                A1_REGISTER_PTR
            } as u32,
        );
        let mut source = read_rv32_register(
            state.memory.data(),
            if shift == 0 {
                A4_REGISTER_PTR
            } else {
                A3_REGISTER_PTR
            } as u32,
        );
        let mut len = read_rv32_register(state.memory.data(), A2_REGISTER_PTR as u32);

        // Store the original values in the record
        record.shift = [shift % 2, shift / 2];
        record.from_pc = *state.pc;
        record.from_timestamp = state.memory.timestamp;

        let num_iterations = (len - shift as u32) & !15;
        let to_dest = dest + num_iterations;
        let to_source = source + num_iterations;
        let to_len = len - num_iterations;

        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            if shift == 0 {
                A3_REGISTER_PTR
            } else {
                A1_REGISTER_PTR
            } as u32,
            to_dest.to_le_bytes(),
            &mut record.register_aux[0].prev_timestamp,
            &mut record.dest,
        );

        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            if shift == 0 {
                A4_REGISTER_PTR
            } else {
                A3_REGISTER_PTR
            } as u32,
            to_source.to_le_bytes(),
            &mut record.register_aux[1].prev_timestamp,
            &mut record.source,
        );

        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            A2_REGISTER_PTR as u32,
            to_len.to_le_bytes(),
            &mut record.register_aux[2].prev_timestamp,
            &mut record.len,
        );

        let mut prev_data = if shift == 0 {
            [0; 4]
        } else {
            source -= 12;
            record
                .read_aux
                .push(MemoryReadAuxRecord { prev_timestamp: 0 });
            let data = tracing_read(
                state.memory,
                RV32_MEMORY_AS,
                source - 4,
                &mut record.read_aux.last_mut().unwrap().prev_timestamp,
            );
            record.memory_read_data.push(data);
            data
        };

        while len - shift as u32 > 15 {
            let writes_data: [[u8; MEMCPY_LOOP_NUM_LIMBS]; 4] = array::from_fn(|i| {
                record
                    .read_aux
                    .push(MemoryReadAuxRecord { prev_timestamp: 0 });
                let data = tracing_read(
                    state.memory,
                    RV32_MEMORY_AS,
                    source + 4 * i as u32,
                    &mut record.read_aux.last_mut().unwrap().prev_timestamp,
                );
                record.memory_read_data.push(data);
                let write_data: [u8; MEMCPY_LOOP_NUM_LIMBS] = array::from_fn(|i| {
                    if i < 4 - shift as usize {
                        data[i + shift as usize]
                    } else {
                        prev_data[i - (4 - shift as usize)]
                    }
                });
                prev_data = data;
                write_data
            });
            writes_data.iter().enumerate().for_each(|(i, write_data)| {
                record.write_aux.push(MemoryWriteBytesAuxRecord {
                    prev_timestamp: 0,
                    prev_data: [0; MEMCPY_LOOP_NUM_LIMBS],
                });
                tracing_write(
                    state.memory,
                    RV32_MEMORY_AS,
                    dest + 4 * i as u32,
                    *write_data,
                    &mut record.write_aux.clone().last_mut().unwrap().prev_timestamp,
                    &mut record.write_aux.clone().last_mut().unwrap().prev_data,
                );
            });
            len -= 16;
            source += 16;
            dest += 16;
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for MemcpyLoopFiller {
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut row: &mut [F]) {
        let record: &MemcpyLoopRecord = unsafe { get_record_from_slice(&mut row, ()) };
        let row: &mut MemcpyLoopCols<F> = row.borrow_mut();

        const NUM_WRITES: u32 = 3;

        let shift = record.shift[0] + record.shift[1] * 2;
        let dest = u32::from_le_bytes(record.dest);
        let source = u32::from_le_bytes(record.source);
        let len = u32::from_le_bytes(record.len);
        let num_copies = (len - shift as u32) & !15;
        let to_dest = dest + num_copies;
        let to_source = source + num_copies;
        let to_len = len - num_copies;
        let timestamp = record.from_timestamp;

        let source_minus_twelve_carry = if shift == 0 {
            F::ZERO
        } else {
            F::from_canonical_u8((source % (1 << 8) < 12) as u8)
        };
        let to_source_minus_twelve_carry = if shift == 0 {
            F::ZERO
        } else {
            F::from_canonical_u8((to_source % (1 << 8) < 12) as u8)
        };

        for ((i, cols), register_aux_record) in row
            .write_aux
            .iter_mut()
            .enumerate()
            .zip(record.register_aux.iter())
        {
            mem_helper.fill(
                register_aux_record.prev_timestamp,
                timestamp + i as u32,
                cols,
            );
        }

        row.source_minus_twelve_carry = source_minus_twelve_carry;
        row.to_source_minus_twelve_carry = to_source_minus_twelve_carry;
        row.to_dest = to_dest.to_le_bytes().map(F::from_canonical_u8);
        row.to_source = to_source.to_le_bytes().map(F::from_canonical_u8);
        row.to_len = F::from_canonical_u32(to_len);
        row.to_timestamp =
            F::from_canonical_u32(timestamp + NUM_WRITES + 2 * num_copies + (shift != 0) as u32);
        row.is_valid = F::ONE;
        row.dest = record.dest.map(F::from_canonical_u8);
        row.source = record.source.map(F::from_canonical_u8);
        row.len = record.len.map(F::from_canonical_u8);
        row.shift = record.shift.map(F::from_canonical_u8);
        row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        row.from_state.pc = F::from_canonical_u32(record.from_pc);

        let word_to_u16 = |data: u32| [data & 0xffff, data >> 16];
        let range_check_data = [
            (word_to_u16(len), false),
            (word_to_u16(dest), true),
            (word_to_u16(source - 12 * (shift != 0) as u32), true),
            (word_to_u16(to_dest), true),
            (word_to_u16(to_source - 12 * (shift != 0) as u32), true),
        ];

        range_check_data.iter().for_each(|(data, is_address)| {
            if *is_address {
                self.range_checker_chip
                    .add_count(data[0] >> 2, 2 * MEMCPY_LOOP_LIMB_BITS - 2)
            } else {
                self.range_checker_chip
                    .add_count(data[0], 2 * MEMCPY_LOOP_LIMB_BITS)
            };
            self.range_checker_chip
                .add_count(data[1], self.pointer_max_bits - 2 * MEMCPY_LOOP_LIMB_BITS);
        });

        // Handle MemcpyIter
        self.memcpy_iter_chip.add_new_loop(
            mem_helper,
            timestamp + NUM_WRITES,
            dest - 16,
            source - 16 - 12 * (shift != 0) as u32,
            len - shift as u32,
            shift,
            record.memory_read_data.clone(),
            record.read_aux.clone(),
            record.write_aux.clone(),
        );
    }
}

impl<F: PrimeField32> Executor<F> for MemcpyLoopExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<MemcpyLoopPreCompute>()
    }

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E1ExecutionCtx,
    {
        let data: &mut MemcpyLoopPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for MemcpyLoopExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MemcpyLoopPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E2ExecutionCtx,
    {
        let data: &mut E2PreCompute<MemcpyLoopPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &MemcpyLoopPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let shift = pre_compute.c;
    let (dest, source) = if shift == 0 {
        (
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A3_REGISTER_PTR as u32),
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A4_REGISTER_PTR as u32),
        )
    } else {
        (
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A1_REGISTER_PTR as u32),
            vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A3_REGISTER_PTR as u32),
        )
    };
    let len = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, A2_REGISTER_PTR as u32);

    let mut dest = u32::from_le_bytes(dest);
    let mut source = u32::from_le_bytes(source);
    let mut len = u32::from_le_bytes(len);

    let mut prev_data = if shift == 0 {
        [0; 4]
    } else {
        source -= 12;
        vm_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source - 4)
    };

    while len - shift as u32 > 15 {
        for i in 0..4 {
            let data = vm_state.vm_read::<u8, 4>(RV32_MEMORY_AS, source + 4 * i);
            let write_data: [u8; 4] = array::from_fn(|i| {
                if i < 4 - shift as usize {
                    data[i + shift as usize]
                } else {
                    prev_data[i - (4 - shift as usize)]
                }
            });
            vm_state.vm_write(RV32_MEMORY_AS, dest + 4 * i, &write_data);
            prev_data = data;
        }
        len -= 16;
        source += 16;
        dest += 16;
    }

    // Write the result back to memory
    if shift == 0 {
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A3_REGISTER_PTR as u32,
            &dest.to_le_bytes(),
        );
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A4_REGISTER_PTR as u32,
            &source.to_le_bytes(),
        );
    } else {
        source += 12;
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A1_REGISTER_PTR as u32,
            &dest.to_le_bytes(),
        );
        vm_state.vm_write(
            RV32_REGISTER_AS,
            A3_REGISTER_PTR as u32,
            &source.to_le_bytes(),
        );
    };
    vm_state.vm_write(RV32_REGISTER_AS, A2_REGISTER_PTR as u32, &len.to_le_bytes());

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MemcpyLoopPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MemcpyLoopPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX>(&pre_compute.data, vm_state);
}

impl MemcpyLoopExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MemcpyLoopPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { opcode, c, .. } = inst;
        let c_u32 = c.as_canonical_u32();
        if ![0, 1, 2, 3].contains(&c_u32) {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = MemcpyLoopPreCompute { c: c_u32 as u8 };
        assert_eq!(*opcode, Rv32MemcpyOpcode::MEMCPY_LOOP.global_opcode());
        Ok(())
    }
}
