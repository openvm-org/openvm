use std::{fmt::Debug, marker::PhantomData};

use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr, InlineRecordShape, Variable};
use rvr_openvm_lift::AirIndex;

use crate::{detect_known_field, format_c_byte_array, KnownField, ModOp};

/// Base trait for zero-sized marker types that identify a field kind (modular
/// or Fp2). Provides the C function prefix and known-field suffix lookup used
/// by all three instruction families (arith, iseq, setup).
pub(crate) trait FieldKind: Clone + Debug + Send + Sync + 'static {
    fn c_prefix() -> &'static str;
    /// Returns the C suffix for a known field, or `None` to fall through to the
    /// generic path.
    fn known_suffix(field: KnownField) -> Option<&'static str>;

    fn inline_record_size(_num_limbs: u32) -> Option<usize> {
        None
    }

    fn emit_inline_record(
        _ctx: &mut dyn ExtEmitCtx,
        _from_pc: u32,
        _local_opcode: u8,
        _num_limbs: u32,
        _chip_idx: Option<AirIndex>,
    ) {
    }
}

/// Extension of [`FieldKind`] for the arithmetic instruction family. Adds the
/// IR opname used by [`FieldArithInstr`].
pub(crate) trait ArithKind: FieldKind {
    fn opname() -> &'static str;
}

/// Extension of [`FieldKind`] for the setup instruction family. Adds the
/// IR opname used by [`FieldSetupInstr`].
pub(crate) trait SetupKind: FieldKind {
    fn opname() -> &'static str;
}

/// Extension of [`FieldKind`] for the IS_EQ instruction family. Adds the
/// IR opname used by [`FieldIsEqInstr`]. Not implemented for Fp2 — fp2 has no IS_EQ.
pub(crate) trait IsEqKind: FieldKind {
    fn opname() -> &'static str;

    fn inline_record_size(_num_limbs: u32) -> Option<usize> {
        None
    }

    fn emit_inline_record(
        _ctx: &mut dyn ExtEmitCtx,
        _from_pc: u32,
        _local_opcode: u8,
        _num_limbs: u32,
        _chip_idx: Option<AirIndex>,
    ) {
    }
}

#[derive(Debug, Clone, Copy)]
struct InlineRecordMeta {
    from_pc: u32,
    local_opcode: u8,
    chip_idx: Option<AirIndex>,
}

/// Generic IR node for field arithmetic (ADD, SUB, MUL, DIV).
/// Use the type aliases [`crate::ModArithInstr`] / [`crate::Fp2ArithInstr`]
/// rather than naming this type directly.
#[derive(Debug, Clone)]
pub(crate) struct FieldArithInstr<K: ArithKind> {
    _kind: PhantomData<K>,
    inline_record: Option<InlineRecordMeta>,
    pub op: ModOp,
    pub rd_reg: Variable,
    pub rs1_reg: Variable,
    pub rs2_reg: Variable,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl<K: ArithKind> FieldArithInstr<K> {
    pub fn new(
        op: ModOp,
        rd_reg: Variable,
        rs1_reg: Variable,
        rs2_reg: Variable,
        num_limbs: u32,
        modulus: Vec<u8>,
    ) -> Self {
        Self {
            _kind: PhantomData,
            inline_record: None,
            op,
            rd_reg,
            rs1_reg,
            rs2_reg,
            num_limbs,
            modulus,
        }
    }

    pub fn with_inline_record(
        mut self,
        from_pc: u32,
        local_opcode: u8,
        chip_idx: Option<AirIndex>,
        enabled: bool,
    ) -> Self {
        self.inline_record = enabled.then_some(InlineRecordMeta {
            from_pc,
            local_opcode,
            chip_idx,
        });
        self
    }
}

/// Generic IR node for field IS_EQ.
/// Use the type alias [`crate::ModIsEqInstr`] rather than naming this type directly.
#[derive(Debug, Clone)]
pub(crate) struct FieldIsEqInstr<K: IsEqKind> {
    _kind: PhantomData<K>,
    inline_record: Option<InlineRecordMeta>,
    pub rd_reg: Variable,
    pub rs1_reg: Variable,
    pub rs2_reg: Variable,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl<K: IsEqKind> FieldIsEqInstr<K> {
    pub fn new(
        rd_reg: Variable,
        rs1_reg: Variable,
        rs2_reg: Variable,
        num_limbs: u32,
        modulus: Vec<u8>,
    ) -> Self {
        Self {
            _kind: PhantomData,
            inline_record: None,
            rd_reg,
            rs1_reg,
            rs2_reg,
            num_limbs,
            modulus,
        }
    }

    pub fn with_inline_record(
        mut self,
        from_pc: u32,
        local_opcode: u8,
        chip_idx: Option<AirIndex>,
        enabled: bool,
    ) -> Self {
        self.inline_record = enabled.then_some(InlineRecordMeta {
            from_pc,
            local_opcode,
            chip_idx,
        });
        self
    }
}

/// Generic IR node for field setup (SETUP_ADDSUB and SETUP_MULDIV).
/// The C function name is derived from `K::c_prefix()` as `rvr_ext_{prefix}_setup`.
/// Use the type aliases [`crate::ModSetupInstr`] / [`crate::Fp2SetupInstr`]
/// rather than naming this type directly.
#[derive(Debug, Clone)]
pub(crate) struct FieldSetupInstr<K: SetupKind> {
    _kind: PhantomData<K>,
    inline_record: Option<InlineRecordMeta>,
    pub rd_reg: Variable,
    pub rs1_reg: Variable,
    pub rs2_reg: Variable,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl<K: SetupKind> FieldSetupInstr<K> {
    pub fn new(
        rd_reg: Variable,
        rs1_reg: Variable,
        rs2_reg: Variable,
        num_limbs: u32,
        modulus: Vec<u8>,
    ) -> Self {
        Self {
            _kind: PhantomData,
            inline_record: None,
            rd_reg,
            rs1_reg,
            rs2_reg,
            num_limbs,
            modulus,
        }
    }

    pub fn with_inline_record(
        mut self,
        from_pc: u32,
        local_opcode: u8,
        chip_idx: Option<AirIndex>,
        enabled: bool,
    ) -> Self {
        self.inline_record = enabled.then_some(InlineRecordMeta {
            from_pc,
            local_opcode,
            chip_idx,
        });
        self
    }
}

impl<K: SetupKind> ExtInstr for FieldSetupInstr<K> {
    fn opname(&self) -> &str {
        K::opname()
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let rd = ctx.read_var(self.rd_reg);
        let num_limbs = format!("{}u", self.num_limbs);
        let mod_literal = format_c_byte_array(&self.modulus);
        let name = format!("rvr_ext_{}_setup", K::c_prefix());
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
        ctx.emit_checked_call(&name, &["state", &rd, &rs1, &rs2, &num_limbs, "mod_"]);
        ctx.write_line("}");
        if let Some(meta) = self.inline_record.filter(|_| ctx.inline_record_enabled()) {
            K::emit_inline_record(
                ctx,
                meta.from_pc,
                meta.local_opcode,
                self.num_limbs,
                meta.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.inline_record.and_then(|_| {
            K::inline_record_size(self.num_limbs)
                .map(|record_size| InlineRecordShape::Custom { record_size })
        })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

impl<K: IsEqKind> ExtInstr for FieldIsEqInstr<K> {
    fn opname(&self) -> &str {
        K::opname()
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let prefix = K::c_prefix();
        let known_suffix = detect_known_field(&self.modulus).and_then(K::known_suffix);
        if let Some(suffix) = known_suffix {
            let name = format!("rvr_ext_{prefix}_iseq_{suffix}");
            let val = ctx.emit_call_expr("bool", &name, &["state", &rs1, &rs2]);
            ctx.write_var(self.rd_reg, &val);
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
            let name = format!("rvr_ext_{prefix}_iseq");
            let num_limbs = format!("{}u", self.num_limbs);
            let val = ctx.emit_call_expr("bool", &name, &["state", &rs1, &rs2, &num_limbs, "mod_"]);
            ctx.write_var(self.rd_reg, &val);
            ctx.write_line("}");
        }
        if let Some(meta) = self.inline_record.filter(|_| ctx.inline_record_enabled()) {
            <K as IsEqKind>::emit_inline_record(
                ctx,
                meta.from_pc,
                meta.local_opcode,
                self.num_limbs,
                meta.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.inline_record.and_then(|_| {
            <K as IsEqKind>::inline_record_size(self.num_limbs)
                .map(|record_size| InlineRecordShape::Custom { record_size })
        })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::WriteUnknown { dst: self.rd_reg }
    }
}

impl<K: ArithKind> ExtInstr for FieldArithInstr<K> {
    fn opname(&self) -> &str {
        K::opname()
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let rd = ctx.read_var(self.rd_reg);
        let op_name = self.op.c_name();
        let prefix = K::c_prefix();
        let known_suffix = detect_known_field(&self.modulus).and_then(K::known_suffix);
        if let Some(suffix) = known_suffix {
            let name = format!("rvr_ext_{prefix}_{op_name}_{suffix}");
            ctx.emit_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
            let name = format!("rvr_ext_{prefix}_{op_name}");
            let num_limbs = format!("{}u", self.num_limbs);
            ctx.emit_call(&name, &["state", &rd, &rs1, &rs2, &num_limbs, "mod_"]);
            ctx.write_line("}");
        }
        if let Some(meta) = self.inline_record.filter(|_| ctx.inline_record_enabled()) {
            K::emit_inline_record(
                ctx,
                meta.from_pc,
                meta.local_opcode,
                self.num_limbs,
                meta.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.inline_record.and_then(|_| {
            K::inline_record_size(self.num_limbs)
                .map(|record_size| InlineRecordShape::Custom { record_size })
        })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}
