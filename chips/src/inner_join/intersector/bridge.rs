use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::IntersectorIoCols, IntersectorAir};

impl IntersectorAir {
    /// Sends interactions required by the IsLessThanTuple SubAir
    /// Sends idx with multiplicity out_mult on the intersector_t2_bus (received by t2_chip)
    ///
    /// Receives idx with multiplicity t1_mult on the t1_intersector_bus (sent by t1_chip)
    /// Receives idx with multiplicity t2_mult on the t2_intersector_bus (sent by t2_chip)
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: IntersectorIoCols<AB::Var>,
    ) {
        builder.push_send(
            self.buses.intersector_t2_bus_index,
            io.idx.clone(),
            io.out_mult,
        );

        builder.push_receive(
            self.buses.t1_intersector_bus_index,
            io.idx.clone(),
            io.t1_mult,
        );
        builder.push_receive(
            self.buses.t2_intersector_bus_index,
            io.idx.clone(),
            io.t2_mult,
        );
    }
}
