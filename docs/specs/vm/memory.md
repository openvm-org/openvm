Any offline memory checking aims to have a transcript consisting of (a,v,t) with address a, value v, timestamp t. The timestamp is per-address. Memory aims to support two operations: read and write.

A read of (a, v) at time t means that if we look through the transcript focusing on only entries with address a, the entry with timestamp immediately preceding t must have value also equal v.
A write of (a, v) at time t means a new entry (a, v, t) must be introduced to the transcript.

The transcript does not need to be materialized explicitly in one place. The particular entries of the transcript are materialized on a per-access basis (in whatever chip needs it). To avoid the materialization, we need a special permutation argument, done via logup (interactions), which allows the transcript to be reflected in virtual buses without being materialized:

We have an offline checking bus, where message fields consist of (a,v,t). The bus then has two sets (send vs receive) that must be equal at the end. In the present case, let us not call the sets "send" vs "receive", instead we will call it "current" and "next" (in the literature it is sometimes called Read and Write sets). Any memory access in a chip must add one entry into each set and constrain a relation between them:

A read of (a, v) at time t must add (a,v,t) to current set and (a,v,t+1) to next set.

This requires two interactions, but the only main trace cells needed are (a,v,t)

A write of (a,v) at time t+1 must add (a,v_prev,t) to current set and (a,v,t+1) to next set, where v_prev is the previous value before the write.

This requires two interactions, and main trace cells for a,v_prev,v,t

The time stamp can only ever increase by 1 to ensure sequential access and avoid any less-than checks.

To balance the current and next sets, some memory chip will need to ensure that every accessed address has an initial (a,v_init,0) added to next set, and (a,v_final,t_last) added to current set.

This last part scales with the total number of addresses that are accessed within the circuit.

In this way, the constraints in each chip's memory accesses creates a "zig-zag" between current and next that both imposes the grouping by