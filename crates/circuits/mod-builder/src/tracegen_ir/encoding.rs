//! Flat `u32` encoding consumed by the CUDA trace-generation interpreter.
//!
//! [`TracegenIr`] is serialized as one self-describing blob:
//!
//! ```text
//! [header | eval ops | witness ops | constraints | field constants | payloads | opcode table]
//! ```
//!
//! The fixed-size header contains dimensions, section counts, and absolute word offsets into the
//! blob. Evaluation operations occupy five words, witness operations occupy nine words, and
//! constraint metadata occupies eight words per constraint. The remaining sections contain
//! variable-length field constants, Montgomery values, signed limb values, and opcode-to-flag
//! mappings. Signed `i32` values are stored by preserving their bit pattern in a `u32`.
//!
//! The constants generated from `tracegen_abi.def` define the header indices and opcode values
//! shared by this encoder and the CUDA decoder, so changing the layout requires updating that ABI.

use super::{abi::*, TracegenIr};

impl TracegenIr {
    /// Encodes this IR as a single `u32` blob with a fixed header and section offsets.
    pub fn encode(&self) -> Vec<u32> {
        let mut blob = vec![0u32; TRACEGEN_HEADER_WORDS as usize];
        blob[H_NUM_LIMBS as usize] = self.num_limbs as u32;
        blob[H_LIMB_BITS as usize] = self.limb_bits as u32;
        blob[H_K as usize] = self.k as u32;
        blob[H_NUM_INPUT as usize] = self.num_input as u32;
        blob[H_NUM_VARS as usize] = self.num_vars as u32;
        blob[H_NUM_FLAGS as usize] = self.num_flags as u32;
        blob[H_NEEDS_SETUP as usize] = self.needs_setup as u32;
        blob[H_WIDTH as usize] = self.width as u32;
        blob[H_NUM_SLOTS as usize] = self.num_eval_slots as u32;
        blob[H_N_EVAL_OPS as usize] = self.eval_ops.len() as u32;
        blob[H_N_WITNESS_OPS as usize] = self.witness_ops.len() as u32;
        blob[H_N_CONS as usize] = self.constraints.len() as u32;
        blob[H_SCRATCH_LEN as usize] = self.scratch_len as u32;
        blob[H_P8_LEN as usize] = self.p8.len() as u32;
        blob[H_N_LOCAL_OPS as usize] = self.local_opcode_idx.len() as u32;
        blob[H_N_OP_FLAGS as usize] = self.opcode_flag_idx.len() as u32;

        blob[H_OFF_EVAL_OPS as usize] = blob.len() as u32;
        for op in &self.eval_ops {
            blob.extend([op.opcode as u32, op.flag, op.dst, op.a, op.b]);
        }
        blob[H_OFF_WITNESS_OPS as usize] = blob.len() as u32;
        for op in &self.witness_ops {
            blob.extend([
                op.opcode as u32,
                op.flag,
                op.dst_off,
                op.dst_len,
                op.a_off,
                op.a_len,
                op.b_off,
                op.b_len,
                op.imm as u32,
            ]);
        }
        blob[H_OFF_CONS as usize] = blob.len() as u32;
        for constraint in &self.constraints {
            blob.extend([
                constraint.tape_start as u32,
                constraint.tape_len as u32,
                constraint.result_off,
                constraint.result_len,
                constraint.q_limbs as u32,
                constraint.carry_limbs as u32,
                constraint.carry_min_abs,
                constraint.carry_bits,
            ]);
        }
        blob[H_OFF_P as usize] = blob.len() as u32;
        blob.extend(&self.p_u32);
        blob[H_OFF_R2 as usize] = blob.len() as u32;
        blob.extend(&self.r2_u32);
        blob[H_OFF_PM2 as usize] = blob.len() as u32;
        blob.extend(&self.pm2_u32);
        blob[H_OFF_PINV as usize] = blob.len() as u32;
        blob.extend(&self.pinv_u32);
        blob[H_OFF_P8 as usize] = blob.len() as u32;
        blob.extend(self.p8.iter().map(|&x| x as u32));
        blob[H_OFF_MONT as usize] = blob.len() as u32;
        blob.extend(&self.mont_payload);
        blob[H_OFF_CLIMBS as usize] = blob.len() as u32;
        blob.extend(self.const_limbs_payload.iter().map(|&x| x as u32));
        blob[H_OFF_OPTAB as usize] = blob.len() as u32;
        blob.extend(self.local_opcode_idx.iter().map(|&x| x as u32));
        blob.extend(self.opcode_flag_idx.iter().map(|&x| x as u32));
        blob[H_MPRIME as usize] = self.mprime;
        blob
    }
}
