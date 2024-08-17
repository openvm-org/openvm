# STARK Architecture

We build our virtual machines in a STARK proving system with a multi-matrix commitment scheme and shared verifier randomness between AIR matrices to enable permutation arguments such as log-up.

In the following, we will refer to a circuit as a collection of AIR matrices (also referred to as chips) of possibly different heights, which may communicate with one another over buses using a log-up permutation argument. We refer to messages sent to such a bus as [interactions](../../../stark-backend/src/interaction/README.md).

Our framework is modular and allows the creation of custom VM circuits to support different instruction sets that follow our overall ISA framework.

## Motivation

We want to make the VM modular, so that adding new instructions and chips involves minimal to no changes to any centralized chip (commonly the CPU chip). We also want to avoid increasing the columns/interactions/buses of the CPU when we add new instructions/chips.

## Design

The following must exist in any VM circuit:

- The program chip
- A program bus
- An execution bus
- A memory bus

Notably, there is no CPU chip where the full transcript of executed instructions is materialized in a single trace matrix. The transcript of memory accesses is also not materialized in a single trace matrix. We discuss reasons for these choices below.

### Program Chip

We follow the Harvard architecture where the program code (ROM) is stored separately from memory. The program chip's trace matrix simply consists of the program code, one instruction per row, as a cached trace, together with interactions on the PROGRAM_BUS.

A cached trace is used so that the commitment to the program code is the proof system trace commitment. This commitment could be changed to a flat hash, likely with worse performance.

### No-CPU

The main motivation is that the existence of a CPU forces the existence of a trace matrix with rows growing with the total number of clock cycles of the program execution. We claim that the no-CPU design gives the minimum lower bound on the number of required trace cells added per opcode execution.

Traditionally, the CPU is in charge of reading/writing from memory and forwarding that information to the appropriate chip. We are changing to a model where each chip directly accesses memory itself. This design was introduced by D. Mittal.

Each chip has IO columns `(timestamp, pc, instruction)` where `instruction` is `(opcode, operands)`.
The chip would receive `(pc, instruction)` on the PROGRAM_BUS to ensure it is reading the correct line of the program code.
There is a maximum length to `operands` defined by the PROGRAM_BUS, but each chip can receive only a subset of the operands (setting the rest to zero) without paying the cost for the unused operands.

Each chip receives `(timestamp, pc)` on EXECUTION_BUS and "after"
executing an instruction, sends `(new_timestamp, new_pc)` on the same bus.
The chip is in charge of constraining that `new_timestamp > timestamp`. (Here we say "after" to correspond to the temporal nature of runtime execution, but there is no before/after in the AIR matrix itself.)
The bus enforces that each timestamp transition corresponds to a particular instruction being executed.

The chip must constrain that `opcode` is one of the opcodes the chip itself owns. The chip then constrains the rest of the validity of the opcode execution, according to the opcode spec.

When there are no continuations, there will need to be another very simple source/sink chip with a 2 row trace that sends out (1, 0) on EXECUTION_BUS and receives `(final_timestamp, final_pc)` on EXECUTION_BUS. With continuations, the start and end timestamp/pc will need to be constrained with respect to the pre/post-states.
