
use std::{borrow::BorrowMut, mem::{align_of, size_of}};

use openvm_circuit::{arch::*, system::{memory::online::TracingMemory, poseidon2::trace}};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_new_keccak256_transpiler::XorinOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{p3_field::PrimeField32, prover::metrics::TraceCells};
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteBytesAuxRecord;

use crate::{keccakf::KeccakfVmExecutor};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use crate::xorin::XorinVmFiller;
use openvm_circuit::system::memory::MemoryAuxColsFactory;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;

use openvm_stark_backend::{
    p3_matrix::{dense::RowMajorMatrix}
};

#[derive(Clone, Copy)]
pub struct KeccakfVmMetadata {
}

impl MultiRowMetadata for KeccakfVmMetadata {
    fn get_num_rows(&self) -> usize {
        1
    }
}

pub(crate) type KeccakfVmRecordLayout = MultiRowLayout<KeccakfVmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct KeccakfVmRecordHeader {
}

pub struct KeccakfVmRecordMut<'a> {
    pub inner: &'a mut KeccakfVmRecordHeader,
}

impl<'a> CustomBorrow<'a, KeccakfVmRecordMut<'a>, KeccakfVmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: KeccakfVmRecordLayout) -> KeccakfVmRecordMut<'a> {
        let (record_buf, _rest) = unsafe {
            self.split_at_mut_unchecked(size_of::<KeccakfVmRecordHeader>())
        };
        KeccakfVmRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> KeccakfVmRecordLayout {
        KeccakfVmRecordLayout {
            metadata: KeccakfVmMetadata {
            },
        }
    }
}

impl<F, RA> PreflightExecutor<F, RA> for KeccakfVmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, KeccakfVmRecordLayout, KeccakfVmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", KeccakfOpcode::KECCAKF)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        todo!()
    }
}
