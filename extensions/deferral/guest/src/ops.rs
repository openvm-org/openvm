#[allow(unused_imports)]
use crate::{
    encode_deferral_imm, Commit, DeferralImmOpcode, OutputKey, COMMIT_NUM_BYTES, DEFERRAL_FUNCT3,
    MAX_DEF_CIRCUITS, OPCODE,
};

/// Macro to generate SETUP opcode
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
    deferral_setup!();
}

/// Execute a deferral call for a compile-time deferral index using raw pointers
#[inline(always)]
#[allow(unused_variables)]
pub fn compute_deferral_output_raw<const DEFERRAL_IDX: u16>(
    output_key_ptr: *mut u8,
    input_commit_ptr: *const u8,
) {
    debug_assert!(DEFERRAL_IDX < MAX_DEF_CIRCUITS);
    deferral_call!(output_key_ptr, input_commit_ptr, DEFERRAL_IDX);
}

/// Retrieve deferral output for a compile-time deferral index using raw pointers
#[inline(always)]
#[allow(unused_variables)]
pub fn get_deferral_output_raw<const DEFERRAL_IDX: u16>(
    output_ptr: *mut u8,
    output_key_ptr: *const u8,
) {
    debug_assert!(DEFERRAL_IDX < MAX_DEF_CIRCUITS);
    deferral_output!(output_ptr, output_key_ptr, DEFERRAL_IDX);
}

/// Execute a deferral call for a compile-time deferral index
#[inline(always)]
pub fn compute_deferral_output<const DEFERRAL_IDX: u16>(input_commit: &Commit) -> OutputKey {
    let mut output_key = OutputKey::new([0u8; COMMIT_NUM_BYTES], 0);
    compute_deferral_output_raw::<DEFERRAL_IDX>(output_key.as_mut_ptr(), input_commit.as_ptr());
    output_key
}

/// Retrieve deferral output for a compile-time deferral index into a caller-provided buffer
#[inline(always)]
pub fn get_deferral_output<const DEFERRAL_IDX: u16>(output: &mut [u8], output_key: &OutputKey) {
    assert_eq!(output.len(), output_key.output_len() as usize);
    get_deferral_output_raw::<DEFERRAL_IDX>(output.as_mut_ptr(), output_key.as_ptr());
}
