#[allow(unused_imports)]
use crate::{
    encode_deferral_imm, Commit, DeferralImmOpcode, OutputKey, COMMIT_NUM_BYTES, DEFERRAL_FUNCT3,
    MAX_DEF_CIRCUITS, OPCODE,
};

/// Macro to generate SETUP opcode
#[cfg(target_os = "zkvm")]
#[macro_export]
macro_rules! deferral_setup {
    () => {
        openvm_custom_insn::custom_insn_i!(
            opcode = OPCODE,
            funct3 = DEFERRAL_FUNCT3,
            rd = Const "x0",
            rs1 = Const "x0",
            imm = Const DeferralImmOpcode::Setup as u16,
        )
    };
}

/// Macro to generate CALL opcode
#[cfg(target_os = "zkvm")]
#[macro_export]
macro_rules! deferral_call {
    ($output_key_ptr:expr, $input_commit_ptr:expr, $deferral_idx:expr) => {{
        openvm_custom_insn::custom_insn_i!(
            opcode = OPCODE,
            funct3 = DEFERRAL_FUNCT3,
            rd = In $output_key_ptr,
            rs1 = In $input_commit_ptr,
            imm = Const encode_deferral_imm($deferral_idx, DeferralImmOpcode::Call),
        )
    }};
}

/// Macro to generate OUTPUT opcode
#[cfg(target_os = "zkvm")]
#[macro_export]
macro_rules! deferral_output {
    ($output_ptr:expr, $output_key_ptr:expr, $deferral_idx:expr) => {{
        openvm_custom_insn::custom_insn_i!(
            opcode = OPCODE,
            funct3 = DEFERRAL_FUNCT3,
            rd = In $output_ptr,
            rs1 = In $output_key_ptr,
            imm = Const encode_deferral_imm($deferral_idx, DeferralImmOpcode::Output),
        )
    }};
}

/// Execute deferral setup
#[inline(always)]
pub fn setup_deferrals() {
    #[cfg(target_os = "zkvm")]
    deferral_setup!();

    #[cfg(not(target_os = "zkvm"))]
    unimplemented!("Deferral framework is only available with zkvm")
}

/// Execute a deferral call for a compile-time deferral index using raw pointers
#[inline(always)]
#[allow(unused_variables)]
pub fn deferred_compute_raw<const DEFERRAL_IDX: u16>(
    output_key_ptr: *mut u8,
    input_commit_ptr: *const u8,
) {
    debug_assert!(DEFERRAL_IDX < MAX_DEF_CIRCUITS);

    #[cfg(target_os = "zkvm")]
    deferral_call!(output_key_ptr, input_commit_ptr, DEFERRAL_IDX);

    #[cfg(not(target_os = "zkvm"))]
    unimplemented!("Deferral framework is only available with zkvm")
}

/// Retrieve deferral output for a compile-time deferral index using raw pointers
#[inline(always)]
#[allow(unused_variables)]
pub fn get_deferred_output_raw<const DEFERRAL_IDX: u16>(
    output_ptr: *mut u8,
    output_key_ptr: *const u8,
) {
    debug_assert!(DEFERRAL_IDX < MAX_DEF_CIRCUITS);

    #[cfg(target_os = "zkvm")]
    deferral_output!(output_ptr, output_key_ptr, DEFERRAL_IDX);

    #[cfg(not(target_os = "zkvm"))]
    unimplemented!("Deferral framework is only available with zkvm")
}

/// Execute a deferral call for a compile-time deferral index
#[inline(always)]
#[allow(unused_variables)]
pub fn deferred_compute<const DEFERRAL_IDX: u16>(input_commit: &Commit) -> OutputKey {
    #[cfg(target_os = "zkvm")]
    let ret = {
        let mut output_key = OutputKey::new([0u8; COMMIT_NUM_BYTES], 0);
        deferred_compute_raw::<DEFERRAL_IDX>(output_key.as_mut_ptr(), input_commit.as_ptr());
        output_key
    };
    #[cfg(target_os = "zkvm")]
    return ret;

    #[cfg(not(target_os = "zkvm"))]
    unimplemented!("Deferral framework is only available with zkvm")
}

/// Retrieve deferral output for a compile-time deferral index into a caller-provided buffer
#[inline(always)]
#[allow(unused_variables)]
pub fn get_deferred_output<const DEFERRAL_IDX: u16>(output: &mut [u8], output_key: &OutputKey) {
    assert_eq!(output.len(), output_key.output_len as usize);

    #[cfg(target_os = "zkvm")]
    get_deferred_output_raw::<DEFERRAL_IDX>(output.as_mut_ptr(), output_key.as_ptr());

    #[cfg(not(target_os = "zkvm"))]
    unimplemented!("Deferral framework is only available with zkvm")
}
