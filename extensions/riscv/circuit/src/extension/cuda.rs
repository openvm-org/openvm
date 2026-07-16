use std::sync::Arc;

use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension,
    },
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    Rv64AddIAir, Rv64AddIChipGpu, Rv64AddIWAir, Rv64AddIWChipGpu, Rv64AddSubAir, Rv64AddSubChipGpu,
    Rv64AddSubWAir, Rv64AddSubWChipGpu, Rv64AuipcAir, Rv64AuipcChipGpu, Rv64BitwiseLogicAir,
    Rv64BitwiseLogicChipGpu, Rv64BitwiseLogicImmAir, Rv64BitwiseLogicImmChipGpu,
    Rv64BranchEqualAir, Rv64BranchEqualChipGpu, Rv64BranchLessThanAir, Rv64BranchLessThanChipGpu,
    Rv64DivRemAir, Rv64DivRemChipGpu, Rv64DivRemWAir, Rv64DivRemWChipGpu, Rv64HintStoreAir,
    Rv64HintStoreChipGpu, Rv64I, Rv64Io, Rv64JalLuiAir, Rv64JalLuiChipGpu, Rv64JalrAir,
    Rv64JalrChipGpu, Rv64LessThanAir, Rv64LessThanChipGpu, Rv64LessThanImmAir,
    Rv64LessThanImmChipGpu, Rv64LoadByteAir, Rv64LoadByteChipGpu, Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChipGpu, Rv64LoadHalfwordAir, Rv64LoadHalfwordChipGpu,
    Rv64LoadSignExtendByteAir, Rv64LoadSignExtendByteChipGpu, Rv64LoadSignExtendHalfwordAir,
    Rv64LoadSignExtendHalfwordChipGpu, Rv64LoadSignExtendWordAir, Rv64LoadSignExtendWordChipGpu,
    Rv64LoadWordAir, Rv64LoadWordChipGpu, Rv64M, Rv64MulHAir, Rv64MulHChipGpu, Rv64MulWAir,
    Rv64MulWChipGpu, Rv64MultiplicationAir, Rv64MultiplicationChipGpu, Rv64ShiftLogicalAir,
    Rv64ShiftLogicalChipGpu, Rv64ShiftLogicalImmAir, Rv64ShiftLogicalImmChipGpu,
    Rv64ShiftRightArithmeticAir, Rv64ShiftRightArithmeticChipGpu, Rv64ShiftRightArithmeticImmAir,
    Rv64ShiftRightArithmeticImmChipGpu, Rv64ShiftWLogicalAir, Rv64ShiftWLogicalChipGpu,
    Rv64ShiftWLogicalImmAir, Rv64ShiftWLogicalImmChipGpu, Rv64ShiftWRightArithmeticAir,
    Rv64ShiftWRightArithmeticChipGpu, Rv64ShiftWRightArithmeticImmAir,
    Rv64ShiftWRightArithmeticImmChipGpu, Rv64StoreByteAir, Rv64StoreByteChipGpu,
    Rv64StoreDoublewordAir, Rv64StoreDoublewordChipGpu, Rv64StoreHalfwordAir,
    Rv64StoreHalfwordChipGpu, Rv64StoreWordAir, Rv64StoreWordChipGpu,
};

/// A `default()` value is correct for configs whose builder does not adopt
/// compact wire arenas (the fresh decode state stays unbound and the chips
/// only ever see expanded records); builders that DO opt into
/// `OPENVM_RVR_GPU_RECORDS=compact` must construct this with their own shared
/// state so the bound operand table reaches the chips.
#[derive(Clone, Default)]
pub struct Rv64ImGpuProverExt {
    /// M-GPUDEC shared decode state, cloned into migrated GPU chips.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: std::sync::Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64I> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64AddSubAir>()?;
        let add_sub = Rv64AddSubChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(add_sub);

        inventory.next_air::<Rv64BitwiseLogicAir>()?;
        let bitwise_logic = Rv64BitwiseLogicChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(bitwise_logic);

        inventory.next_air::<Rv64AddSubWAir>()?;
        let add_sub_w = Rv64AddSubWChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(add_sub_w);

        inventory.next_air::<Rv64LessThanAir>()?;
        let lt = Rv64LessThanChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64ShiftLogicalAir>()?;
        let shift_logical = Rv64ShiftLogicalChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(shift_logical);

        inventory.next_air::<Rv64ShiftRightArithmeticAir>()?;
        let shift_right_arithmetic = Rv64ShiftRightArithmeticChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(shift_right_arithmetic);

        inventory.next_air::<Rv64ShiftWLogicalAir>()?;
        let shift_w_logical = Rv64ShiftWLogicalChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(shift_w_logical);

        inventory.next_air::<Rv64ShiftWRightArithmeticAir>()?;
        let shift_w_right_arithmetic = Rv64ShiftWRightArithmeticChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(shift_w_right_arithmetic);

        inventory.next_air::<Rv64AddIWAir>()?;
        let addi_w = Rv64AddIWChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(addi_w);

        inventory.next_air::<Rv64ShiftWLogicalImmAir>()?;
        let shift_w_logical_imm =
            Rv64ShiftWLogicalImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_logical_imm);

        inventory.next_air::<Rv64ShiftWRightArithmeticImmAir>()?;
        let shift_w_right_arithmetic_imm =
            Rv64ShiftWRightArithmeticImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_w_right_arithmetic_imm);

        inventory.next_air::<Rv64LoadSignExtendByteAir>()?;
        let load_sign_extend_byte = Rv64LoadSignExtendByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_sign_extend_byte);

        inventory.next_air::<Rv64LoadByteAir>()?;
        let load_byte = Rv64LoadByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_byte);

        inventory.next_air::<Rv64StoreByteAir>()?;
        let store_byte = Rv64StoreByteChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(store_byte);

        inventory.next_air::<Rv64LoadSignExtendHalfwordAir>()?;
        let load_sign_extend_halfword = Rv64LoadSignExtendHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_sign_extend_halfword);

        inventory.next_air::<Rv64LoadHalfwordAir>()?;
        let load_halfword = Rv64LoadHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_halfword);

        inventory.next_air::<Rv64StoreHalfwordAir>()?;
        let store_halfword = Rv64StoreHalfwordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(store_halfword);

        inventory.next_air::<Rv64LoadSignExtendWordAir>()?;
        let load_sign_extend_word = Rv64LoadSignExtendWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_sign_extend_word);

        inventory.next_air::<Rv64LoadWordAir>()?;
        let load_word = Rv64LoadWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_word);

        inventory.next_air::<Rv64StoreWordAir>()?;
        let store_word = Rv64StoreWordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(store_word);

        inventory.next_air::<Rv64LoadDoublewordAir>()?;
        let load_doubleword = Rv64LoadDoublewordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(load_doubleword);

        inventory.next_air::<Rv64StoreDoublewordAir>()?;
        let store_doubleword = Rv64StoreDoublewordChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(store_doubleword);

        inventory.next_air::<Rv64BranchEqualAir>()?;
        let beq = Rv64BranchEqualChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThanAir>()?;
        let blt = Rv64BranchLessThanChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64JalLuiAir>()?;
        let jal_lui = Rv64JalLuiChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv64JalrAir>()?;
        let jalr = Rv64JalrChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv64AuipcAir>()?;
        let auipc = Rv64AuipcChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(auipc);

        inventory.next_air::<Rv64AddIAir>()?;
        let addi = Rv64AddIChipGpu::new(
            range_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(addi);

        inventory.next_air::<Rv64ShiftLogicalImmAir>()?;
        let shift_logical_imm =
            Rv64ShiftLogicalImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_logical_imm);

        inventory.next_air::<Rv64ShiftRightArithmeticImmAir>()?;
        let shift_right_arithmetic_imm =
            Rv64ShiftRightArithmeticImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(shift_right_arithmetic_imm);

        inventory.next_air::<Rv64LessThanImmAir>()?;
        let lt_imm = Rv64LessThanImmChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(lt_imm);

        inventory.next_air::<Rv64BitwiseLogicImmAir>()?;
        let bitwise_logic_imm = Rv64BitwiseLogicImmChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(bitwise_logic_imm);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64M> for Rv64ImGpuProverExt {
    fn extend_prover(
        &self,
        extension: &Rv64M,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                    range_checker.device_ctx.clone(),
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64MultiplicationAir>()?;
        let mult = Rv64MultiplicationChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64MulWAir>()?;
        let mul_w = Rv64MulWChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(mul_w);

        inventory.next_air::<Rv64MulHAir>()?;
        let mul_h = Rv64MulHChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv64DivRemAir>()?;
        let div_rem = Rv64DivRemChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(div_rem);

        inventory.next_air::<Rv64DivRemWAir>()?;
        let divrem_w = Rv64DivRemWChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(divrem_w);

        Ok(())
    }
}

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv64Io>
    for Rv64ImGpuProverExt
{
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);

        inventory.next_air::<Rv64HintStoreAir>()?;
        let hint_store = Rv64HintStoreChipGpu::new(
            range_checker.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits,
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            self.rvr_decode.clone(),
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}
