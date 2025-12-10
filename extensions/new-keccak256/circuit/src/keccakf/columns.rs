use core::mem::size_of;

use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_air::AirBuilder;
