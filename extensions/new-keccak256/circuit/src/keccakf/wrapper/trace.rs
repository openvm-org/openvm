
use std::{borrow::BorrowMut, mem::{align_of, size_of}};
use core::convert::TryInto;

use openvm_circuit::{arch::*, system::{memory::online::TracingMemory, poseidon2::trace}};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{p3_field::PrimeField32, prover::metrics::TraceCells};
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteBytesAuxRecord;
use openvm_instructions::riscv::{RV32_REGISTER_AS, RV32_MEMORY_AS};

use crate::keccakf::{KeccakfVmExecutor, wrapper::columns::NUM_KECCAKF_WRAPPER_COLS};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use crate::keccakf::KeccakfVmFiller;
use openvm_circuit::system::memory::MemoryAuxColsFactory;
use crate::keccakf::columns::KeccakfVmCols;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_instructions::riscv::RV32_CELL_BITS;
use p3_keccak_air::generate_trace_rows;
use openvm_stark_backend::{
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
};
use crate::keccakf::wrapper::KeccakfWrapperFiller;

impl<F: PrimeField32> TraceFiller<F> for KeccakfWrapperFiller {
    fn fill_trace_row(
        &self,
        _mem_helper: &MemoryAuxColsFactory<F>,
        row_slice: &mut [F],
    ) {
        let p3_trace: RowMajorMatrix<F> = generate_trace_rows(vec![[0u64; 25]; 1], 0);
        row_slice[..NUM_KECCAKF_WRAPPER_COLS].copy_from_slice(
            &p3_trace.values[..NUM_KECCAKF_WRAPPER_COLS]
        );
    }
}