use std::{
    borrow::{Borrow, BorrowMut},
    convert::TryInto,
    mem::size_of,
    sync::OnceLock,
};

use openvm_circuit::{
    arch::{StaticProgramError, *},
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_poseidon2_air::{
    p3_baby_bear::BabyBear, Poseidon2Config, Poseidon2SubChip, POSEIDON2_WIDTH,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{Poseidon2PermuteExecutor, NUM_OP_ROWS_PER_INS};
use crate::{POSEIDON2_STATE_BYTES, POSEIDON2_WORD_SIZE, SBOX_REGISTERS};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Poseidon2PermutePreCompute {
    a: u8,
}

/// Recomposes the 16 little-endian u32 words of the preimage into field elements.
pub(super) fn recompose_words<F: PrimeField32>(
    preimage: &[u8; POSEIDON2_STATE_BYTES],
) -> [F; POSEIDON2_WIDTH] {
    std::array::from_fn(|i| {
        let word = u32::from_le_bytes(preimage[i * 4..i * 4 + 4].try_into().unwrap());
        F::from_u32(word)
    })
}

/// Decomposes the 16 field-element words of the postimage into little-endian bytes.
pub(super) fn decompose_bytes<F: PrimeField32>(
    postimage: [F; POSEIDON2_WIDTH],
) -> [u8; POSEIDON2_STATE_BYTES] {
    let mut bytes = [0u8; POSEIDON2_STATE_BYTES];
    for (i, word) in postimage.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&word.as_canonical_u32().to_le_bytes());
    }
    bytes
}

/// Computes the permutation of the preimage state on the host.
///
/// INVARIANT: this host-side permutation and the periphery AIR must use the SAME constants. Both
/// are constructed from `Poseidon2Config::default()`; the extension wiring must construct the
/// periphery chip/air exclusively via the [`crate::periphery::poseidon2_periphery_chip`] /
/// [`crate::periphery::poseidon2_periphery_air`] factories (which also use the default config) so
/// the two can never diverge. If the periphery were ever configured with non-default constants,
/// host execution would silently disagree with the AIR and verification would fail.
///
/// The subchip is built once per process and shared by every caller: both the `#[create_handler]`
/// interpreter path (which has no access to executor state) and [`PreflightExecutor::execute`],
/// which runs once per `PERMUTE` instruction on the proving hot path. Building it is not cheap
/// relative to the permutation it performs — `Poseidon2SubChip::new` allocates an
/// `Arc<Poseidon2SubAir>` and the `Vec`s returned by `to_external_internal_constants` — so it must
/// not happen per instruction.
///
/// [`PreflightExecutor::execute`]: openvm_circuit::arch::PreflightExecutor::execute
pub(super) fn poseidon2_permute_bytes<F: VmField>(
    preimage: &[u8; POSEIDON2_STATE_BYTES],
) -> [u8; POSEIDON2_STATE_BYTES] {
    debug_assert_eq!(
        F::ORDER_U32,
        BabyBear::ORDER_U32,
        "poseidon2 round constants are BabyBear-specific, so F must be BabyBear"
    );
    decompose_bytes(host_permuter().permute(recompose_words::<BabyBear>(preimage)))
}

/// The permutation is computed in BabyBear regardless of `F`: `Poseidon2Config::default()` is built
/// from `default_baby_bear_rc()`, so BabyBear is the only field for which the host result matches
/// the periphery AIR. This mirrors `openvm_circuit::arch::hasher::poseidon2::vm_poseidon2_hasher`,
/// which pins the same assumption. Fixing the field is also what lets the subchip be cached in a
/// `static`.
fn host_permuter() -> &'static Poseidon2SubChip<BabyBear, SBOX_REGISTERS> {
    static HOST_PERMUTER: OnceLock<Poseidon2SubChip<BabyBear, SBOX_REGISTERS>> = OnceLock::new();
    HOST_PERMUTER.get_or_init(|| Poseidon2SubChip::new(Poseidon2Config::default().constants))
}

impl Poseidon2PermuteExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Poseidon2PermutePreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode: _,
            a,
            b: _,
            c: _,
            d,
            e,
            ..
        } = inst;

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = Poseidon2PermutePreCompute {
            a: a.as_canonical_u32() as u8,
        };

        Ok(())
    }
}

impl<F: VmField> InterpreterExecutor<F> for Poseidon2PermuteExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<Poseidon2PermutePreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut Poseidon2PermutePreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut Poseidon2PermutePreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: VmField> AotExecutor<F> for Poseidon2PermuteExecutor {}

impl<F: VmField> InterpreterMeteredExecutor<F> for Poseidon2PermuteExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Poseidon2PermutePreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<Poseidon2PermutePreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<Poseidon2PermutePreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

#[cfg(feature = "aot")]
impl<F: VmField> AotMeteredExecutor<F> for Poseidon2PermuteExecutor {}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: VmField, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Poseidon2PermutePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Poseidon2PermutePreCompute>()).borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: VmField, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &Poseidon2PermutePreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rd_ptr = pre_compute.a as u32;
    let buffer_ptr_limbs: [u8; 4] = exec_state.vm_read(RV32_REGISTER_AS, rd_ptr);
    let buffer_ptr = u32::from_le_bytes(buffer_ptr_limbs);

    let preimage: &[u8] =
        exec_state.host_read_slice(RV32_MEMORY_AS, buffer_ptr, POSEIDON2_STATE_BYTES);
    let postimage = poseidon2_permute_bytes::<F>(preimage.try_into().unwrap());

    if IS_E1 {
        exec_state.vm_write(RV32_MEMORY_AS, buffer_ptr, &postimage);
    } else {
        for (word_idx, word) in postimage.chunks_exact(POSEIDON2_WORD_SIZE).enumerate() {
            exec_state.vm_write::<u8, POSEIDON2_WORD_SIZE>(
                RV32_MEMORY_AS,
                buffer_ptr + (word_idx * POSEIDON2_WORD_SIZE) as u32,
                word.try_into().unwrap(),
            );
        }
    }

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: VmField, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Poseidon2PermutePreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Poseidon2PermutePreCompute>>(),
    )
    .borrow();

    let op_air_idx = pre_compute.chip_idx as usize;

    // Update Poseidon2PermuteChip height (1 row per instruction)
    exec_state
        .ctx
        .on_height_change(op_air_idx, NUM_OP_ROWS_PER_INS as u32);

    // HACK: Poseidon2PeripheryAir is added right before Poseidon2PermuteAir in extend_circuit,
    // and due to reverse ordering of AIR indices, periphery_air_idx = op_air_idx + 1.
    // See extension/mod.rs extend_circuit for the ordering.
    let periphery_air_idx = op_air_idx + 1;

    // Update Poseidon2PeripheryChip height (1 row per permutation)
    exec_state.ctx.on_height_change(periphery_air_idx, 1);

    execute_e12_impl::<F, CTX, false>(&pre_compute.data, exec_state);
}
