use crate::{MemWidth, ValueSlot};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedTraceRows {
    pub chip_idx: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageAddressSpace {
    MainMemory(u32),
    Other(u32),
}

impl PageAddressSpace {
    pub const fn id(self) -> u32 {
        match self {
            Self::MainMemory(id) | Self::Other(id) => id,
        }
    }

    pub const fn is_main_memory(self) -> bool {
        matches!(self, Self::MainMemory(_))
    }
}

/// Mode-aware C emission interface available to instruction nodes.
pub trait EmitCtx {
    fn read_slot(&mut self, slot: ValueSlot) -> String;
    fn peek_slot(&mut self, slot: ValueSlot) -> String;
    fn write_slot(&mut self, slot: ValueSlot, val: &str);
    fn write_line(&mut self, s: &str);
    fn emit_trap(&mut self);
    fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String;
    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8);
    fn emit_call(&mut self, name: &str, args: &[&str]);
    fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]);
    fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String;
    fn emit_call_with_trace_result(
        &mut self,
        ret_ty: &str,
        name: &str,
        args: &[&str],
    ) -> Option<String>;

    fn emit_checked_call(&mut self, name: &str, args: &[&str]) {
        let result = self.emit_call_expr("bool", name, args);
        self.write_line(&format!("if (unlikely(!{result})) {{"));
        self.emit_trap();
        self.write_line("}");
    }

    fn emit_checked_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        self.write_line(&format!("if (unlikely(!{name}({}))) {{", args.join(", ")));
        self.emit_trap();
        self.write_line("}");
    }

    fn trace_chip(&mut self, chip_idx: u32, count_expr: &str);
    fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str);
    fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace);
    fn trace_page_access_u64_range(
        &mut self,
        base_addr: &str,
        num_dwords: &str,
        addr_space: PageAddressSpace,
    );
}

pub use EmitCtx as ExtEmitCtx;

pub use crate::Instr as ExtInstr;
