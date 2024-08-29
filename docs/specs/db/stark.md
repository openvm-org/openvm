# STARK Architecture

We describe our design for how to build a verifiable data system using STARKs.

Our data system follows a relational database model where the underlying logical data is stored
in tables with fixed schemas which determine the column structure, while the number of rows
is unbounded.

Terminology:

- **Logical Table**: The table with typed columns as appears in a normal database. The table has a fixed schema and underlying data stored in a variable unbounded number of rows.
- **Logical Page**: A logical table where the number of rows does not exceed some maximum limit set by the system (e.g., around 1 million). The columns of the table are still typed. [This corresponds to `RecordBatch` type in Datafusion.]
- **Physical Page**: <!--[jpw] not sure this is a good name--> The representation of a logical page as a trace matrix over a small prime field. A typed column in logical page may be transformed to multiple columns in the trace matrix (e.g., we default to storing 2 bytes = 16 bits per column).
  The physical page comes with a physical schema, which is deterministically derived from the logical schema. The physical schema specifies the mapping from the logical columns to the physical columns in the trace matrix as well as range guarantees (in bits) for each physical column. At present we will derive physical schema from logical schema without additional configuration parameters. The trace matrix is resized to have height a power of 2 by padding with **unallocated** rows, always at the bottom of the matrix.
- **Physical Dataframe**: An ordered list of physical pages. The list may be unbounded in length.
  The physical dataframe is used to represent a logical table as a sequence of trace matrices over a
  small prime field.
