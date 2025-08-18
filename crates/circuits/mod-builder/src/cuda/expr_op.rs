use crate::ExprNode;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExprOp(pub u128);

impl ExprOp {
    /// Encode an ExprNode into a 128-bit word:
    /// bits [0..8)   = type
    /// bits [8..40)  = data[0]
    /// bits [40..72) = data[1]
    /// bits [72..104)= data[2]
    pub fn from_node(node: &ExprNode) -> Self {
        let ty = (node.r#type as u128) & 0xFF;
        let d0 = (node.data[0] as u128) & 0xFFFFFFFF;
        let d1 = (node.data[1] as u128) & 0xFFFFFFFF;
        let d2 = (node.data[2] as u128) & 0xFFFFFFFF;
        let raw = (ty) | (d0 << 8) | (d1 << 40) | (d2 << 72);
        ExprOp(raw)
    }
}
