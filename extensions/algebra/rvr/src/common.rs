use std::{fmt::Debug, marker::PhantomData};

use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Reg};

use crate::{detect_known_field, format_c_byte_array, KnownField, ModOp};

/// Marker trait implemented by zero-sized types that parameterize
/// [`FieldArithInstr`], supplying the C prefix and known-field suffix lookup.
pub trait ArithKind: Clone + Debug + Send + Sync + 'static {
    fn opname() -> &'static str;
    fn c_prefix() -> &'static str;
    /// Returns the C suffix for a known field, or `None` to fall through to the
    /// generic path.
    fn known_suffix(field: KnownField) -> Option<&'static str>;
}

/// Generic IR node for field arithmetic (ADD, SUB, MUL, DIV).
/// Use the type aliases [`crate::ModArithInstr`] / [`crate::Fp2ArithInstr`]
/// rather than naming this type directly.
#[derive(Debug, Clone)]
pub struct FieldArithInstr<K: ArithKind> {
    _kind: PhantomData<K>,
    pub op: ModOp,
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl<K: ArithKind> FieldArithInstr<K> {
    pub fn new(
        op: ModOp,
        rd_reg: Reg,
        rs1_reg: Reg,
        rs2_reg: Reg,
        num_limbs: u32,
        modulus: Vec<u8>,
    ) -> Self {
        Self { _kind: PhantomData, op, rd_reg, rs1_reg, rs2_reg, num_limbs, modulus }
    }
}

/// Generic IR node for field setup (SETUP_ADDSUB, SETUP_MULDIV, SETUP_ISEQ).
/// Use the type aliases [`crate::ModSetupInstr`] / [`crate::Fp2SetupInstr`]
/// rather than naming this type directly.
#[derive(Debug, Clone)]
pub struct FieldSetupInstr {
    c_fn: &'static str,
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
}

impl FieldSetupInstr {
    pub fn new(c_fn: &'static str, rd_reg: Reg, rs1_reg: Reg, rs2_reg: Reg, num_limbs: u32) -> Self {
        Self { c_fn, rd_reg, rs1_reg, rs2_reg, num_limbs }
    }
}

impl ExtInstr for FieldSetupInstr {
    fn opname(&self) -> &str {
        &self.c_fn["rvr_ext_".len()..]
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.extern_call(self.c_fn, &["state", &rd, &rs1, &rs2, &num_limbs]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

impl<K: ArithKind> ExtInstr for FieldArithInstr<K> {
    fn opname(&self) -> &str {
        K::opname()
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let op_name = self.op.c_name();
        let prefix = K::c_prefix();
        let known_suffix = detect_known_field(&self.modulus).and_then(K::known_suffix);
        if let Some(suffix) = known_suffix {
            let name = format!("rvr_ext_{prefix}_{op_name}_{suffix}");
            ctx.extern_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            let name = format!("rvr_ext_{prefix}_{op_name}");
            let num_limbs = format!("{}u", self.num_limbs);
            ctx.extern_call(&name, &["state", &rd, &rs1, &rs2, &num_limbs, "mod_"]);
            ctx.write_line("}");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}
