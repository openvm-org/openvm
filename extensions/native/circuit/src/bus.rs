use openvm_stark_backend::interaction::{InteractionBuilder, InteractionType};

use crate::air::VerifyBatchBus;

#[derive(Clone, Copy, Debug)]
pub struct OpenedValueBus(pub usize);

impl OpenedValueBus {
    pub fn new(bus: usize) -> Self {
        Self(bus)
    }
    pub fn interact<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        send: bool,
        multiplicity: impl Into<AB::Expr>,
        hint_id: impl Into<AB::Expr>,
        idx: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
    ) {
        let fields = vec![hint_id.into(), idx.into(), value.into()];
        builder.push_interaction(
            self.0,
            fields,
            multiplicity.into(),
            if send {
                InteractionType::Send
            } else {
                InteractionType::Receive
            },
        );
    }
}
