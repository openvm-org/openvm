| Summary | Proof Time (s) | Parallel Proof Time (s) |
|:---|---:|---:|
| Total | <span style='color: red'>(+4 [+68.2%])</span> 10.99 | <span style='color: red'>(+4 [+68.2%])</span> 10.99 |
| fibonacci_program | <span style='color: red'>(+4 [+68.2%])</span> 10.99 | <span style='color: red'>(+4 [+68.2%])</span> 10.99 |


| fibonacci_program |||||
|:---|---:|---:|---:|---:|
|metric|avg|sum|max|min|
| `total_proof_time_ms ` | <span style='color: red'>(+4455 [+68.2%])</span> 10,987 | <span style='color: red'>(+4455 [+68.2%])</span> 10,987 | <span style='color: red'>(+4455 [+68.2%])</span> 10,987 | <span style='color: red'>(+4455 [+68.2%])</span> 10,987 |
| `main_cells_used     ` |  51,503,940 |  51,503,940 |  51,503,940 |  51,503,940 |
| `total_cycles        ` |  1,500,137 |  1,500,137 |  1,500,137 |  1,500,137 |
| `execute_time_ms     ` | <span style='color: red'>(+4670 [+617.7%])</span> 5,426 | <span style='color: red'>(+4670 [+617.7%])</span> 5,426 | <span style='color: red'>(+4670 [+617.7%])</span> 5,426 | <span style='color: red'>(+4670 [+617.7%])</span> 5,426 |
| `trace_gen_time_ms   ` | <span style='color: green'>(-6 [-2.3%])</span> 251 | <span style='color: green'>(-6 [-2.3%])</span> 251 | <span style='color: green'>(-6 [-2.3%])</span> 251 | <span style='color: green'>(-6 [-2.3%])</span> 251 |
| `stark_prove_excluding_trace_time_ms` | <span style='color: green'>(-209 [-3.8%])</span> 5,310 | <span style='color: green'>(-209 [-3.8%])</span> 5,310 | <span style='color: green'>(-209 [-3.8%])</span> 5,310 | <span style='color: green'>(-209 [-3.8%])</span> 5,310 |
| `main_trace_commit_time_ms` | <span style='color: green'>(-59 [-6.9%])</span> 794 | <span style='color: green'>(-59 [-6.9%])</span> 794 | <span style='color: green'>(-59 [-6.9%])</span> 794 | <span style='color: green'>(-59 [-6.9%])</span> 794 |
| `generate_perm_trace_time_ms` | <span style='color: red'>(+1 [+0.6%])</span> 179 | <span style='color: red'>(+1 [+0.6%])</span> 179 | <span style='color: red'>(+1 [+0.6%])</span> 179 | <span style='color: red'>(+1 [+0.6%])</span> 179 |
| `perm_trace_commit_time_ms` | <span style='color: green'>(-139 [-7.9%])</span> 1,611 | <span style='color: green'>(-139 [-7.9%])</span> 1,611 | <span style='color: green'>(-139 [-7.9%])</span> 1,611 | <span style='color: green'>(-139 [-7.9%])</span> 1,611 |
| `quotient_poly_compute_time_ms` | <span style='color: green'>(-3 [-0.4%])</span> 843 | <span style='color: green'>(-3 [-0.4%])</span> 843 | <span style='color: green'>(-3 [-0.4%])</span> 843 | <span style='color: green'>(-3 [-0.4%])</span> 843 |
| `quotient_poly_commit_time_ms` | <span style='color: green'>(-21 [-4.2%])</span> 474 | <span style='color: green'>(-21 [-4.2%])</span> 474 | <span style='color: green'>(-21 [-4.2%])</span> 474 | <span style='color: green'>(-21 [-4.2%])</span> 474 |
| `pcs_opening_time_ms ` | <span style='color: red'>(+14 [+1.0%])</span> 1,407 | <span style='color: red'>(+14 [+1.0%])</span> 1,407 | <span style='color: red'>(+14 [+1.0%])</span> 1,407 | <span style='color: red'>(+14 [+1.0%])</span> 1,407 |



<details>
<summary>Detailed Metrics</summary>

| group | num_segments | keygen_time_ms | commit_exe_time_ms |
| --- | --- | --- | --- |
| fibonacci_program | 1 | 354 | 6 | 

| group | air_name | quotient_deg | interactions | constraints |
| --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<16> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<2> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<32> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<4> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<64> | 2 | 5 | 14 | 
| fibonacci_program | AccessAdapterAir<8> | 2 | 5 | 14 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 2 | 2 | 4 | 
| fibonacci_program | MemoryMerkleAir<8> | 2 | 4 | 40 | 
| fibonacci_program | PersistentBoundaryAir<8> | 2 | 3 | 6 | 
| fibonacci_program | PhantomAir | 2 | 3 | 5 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 2 | 1 | 286 | 
| fibonacci_program | ProgramAir | 1 | 1 | 4 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 1 | 1 | 4 | 
| fibonacci_program | VariableRangeCheckerAir | 1 | 1 | 4 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 2 | 19 | 43 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 2 | 17 | 39 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 2 | 23 | 90 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 2 | 11 | 25 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 2 | 13 | 41 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 2 | 10 | 22 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 2 | 15 | 17 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 2 | 16 | 20 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadSignExtendCoreAir<4, 8> | 2 | 18 | 33 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 2 | 17 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<4, 8> | 2 | 25 | 88 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<4, 8> | 2 | 24 | 38 | 
| fibonacci_program | VmAirWrapper<Rv32MultAdapterAir, MultiplicationCoreAir<4, 8> | 2 | 19 | 26 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 2 | 11 | 15 | 
| fibonacci_program | VmConnectorAir | 2 | 3 | 9 | 

| group | air_name | cycle_tracker_span | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  |  | ADD | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start |  | ADD | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main |  | ADD | 0 | 32,400,684 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main |  | OR | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | ADD | 0 | 252 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | ADD | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | SUB | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | XOR | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | ADD | 0 | 324 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | AND | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | SUB | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | ADD | 0 | 216 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | SUB | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | __start;main |  | SLTU | 0 | 11,100,000 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | SLTU | 0 | 37 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | SLTU | 0 | 37 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | SLL | 0 | 106 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | __start;main |  | BEQ | 0 | 2,600,026 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | __start;main |  | BNE | 0 | 2,600,052 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | BEQ | 0 | 26 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | BNE | 0 | 26 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | BEQ | 0 | 52 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | BNE | 0 | 52 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | BGEU | 0 | 32 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | BLTU | 0 | 64 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | BGEU | 0 | 64 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | __start;main |  | JAL | 0 | 1,800,018 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | __start;main |  | LUI | 0 | 18 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | LUI | 0 | 18 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | LUI | 0 | 18 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LUI | 0 | 72 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | LUI | 0 | 36 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | HINT_STOREW | 0 | 26 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | HINT_STOREW | 0 | 52 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  |  | JALR | 0 | 28 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | __start |  | JALR | 0 | 28 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | __start;main |  | JALR | 0 | 112 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | JALR | 0 | 84 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | JALR | 0 | 28 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | JALR | 0 | 28 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | JALR | 0 | 56 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  |  | LOADW | 0 | 40 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start |  | STOREW | 0 | 40 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main |  | LOADW | 0 | 280 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main |  | STOREW | 0 | 320 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | LOADW | 0 | 80 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | STOREW | 0 | 40 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LOADW | 0 | 40 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | STOREW | 0 | 40 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | LOADW | 0 | 80 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | STOREW | 0 | 160 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  |  | AUIPC | 0 | 42 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | __start |  | AUIPC | 0 | 21 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | __start;main |  | AUIPC | 0 | 63 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | AUIPC | 0 | 42 | 
| fibonacci_program | AccessAdapter<8> |  |  | AUIPC | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> |  |  | LOADW | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start |  | STOREW | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main |  | ADD | 0 | 34 | 
| fibonacci_program | AccessAdapter<8> | __start;main |  | SLTU | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main |  | STOREW | 0 | 85 | 
| fibonacci_program | AccessAdapter<8> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | ADD | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | STOREW | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | ADD | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LOADW | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | SLTU | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | STOREW | 0 | 17 | 
| fibonacci_program | Boundary |  |  | AUIPC | 0 | 40 | 
| fibonacci_program | Boundary |  |  | LOADW | 0 | 40 | 
| fibonacci_program | Boundary | __start |  | STOREW | 0 | 40 | 
| fibonacci_program | Boundary | __start;main |  | ADD | 0 | 80 | 
| fibonacci_program | Boundary | __start;main |  | SLTU | 0 | 40 | 
| fibonacci_program | Boundary | __start;main |  | STOREW | 0 | 200 | 
| fibonacci_program | Boundary | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | ADD | 0 | 40 | 
| fibonacci_program | Boundary | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | STOREW | 0 | 40 | 
| fibonacci_program | Boundary | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | ADD | 0 | 40 | 
| fibonacci_program | Boundary | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LOADW | 0 | 40 | 
| fibonacci_program | Boundary | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | SLTU | 0 | 40 | 
| fibonacci_program | Boundary | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | STOREW | 0 | 40 | 
| fibonacci_program | Merkle |  |  | LOADW | 0 | 1,664 | 
| fibonacci_program | Merkle | __start |  | STOREW | 0 | 704 | 
| fibonacci_program | Merkle | __start;main |  | ADD | 0 | 256 | 
| fibonacci_program | Merkle | __start;main |  | STOREW | 0 | 1,984 | 
| fibonacci_program | Merkle | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | STOREW | 0 | 128 | 
| fibonacci_program | Merkle | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | ADD | 0 | 64 | 
| fibonacci_program | Merkle | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LOADW | 0 | 640 | 
| fibonacci_program | PhantomAir | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | PHANTOM | 0 | 12 | 

| group | air_name | dsl_ir | opcode | segment | cells_used |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | ADD | 0 | 32,401,620 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | AND | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | OR | 0 | 36 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | SUB | 0 | 144 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> |  | XOR | 0 | 72 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> |  | SLTU | 0 | 11,100,074 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> |  | SLL | 0 | 106 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BEQ | 0 | 2,600,104 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> |  | BNE | 0 | 2,600,130 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BGEU | 0 | 96 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> |  | BLTU | 0 | 64 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | JAL | 0 | 1,800,018 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> |  | LUI | 0 | 162 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> |  | HINT_STOREW | 0 | 78 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> |  | JALR | 0 | 364 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | LOADW | 0 | 520 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> |  | STOREW | 0 | 600 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> |  | AUIPC | 0 | 168 | 
| fibonacci_program | AccessAdapter<8> |  | ADD | 0 | 68 | 
| fibonacci_program | AccessAdapter<8> |  | AUIPC | 0 | 17 | 
| fibonacci_program | AccessAdapter<8> |  | LOADW | 0 | 34 | 
| fibonacci_program | AccessAdapter<8> |  | SLTU | 0 | 34 | 
| fibonacci_program | AccessAdapter<8> |  | STOREW | 0 | 136 | 
| fibonacci_program | Boundary |  | ADD | 0 | 160 | 
| fibonacci_program | Boundary |  | AUIPC | 0 | 40 | 
| fibonacci_program | Boundary |  | LOADW | 0 | 80 | 
| fibonacci_program | Boundary |  | SLTU | 0 | 80 | 
| fibonacci_program | Boundary |  | STOREW | 0 | 320 | 
| fibonacci_program | Merkle |  | ADD | 0 | 320 | 
| fibonacci_program | Merkle |  | LOADW | 0 | 2,304 | 
| fibonacci_program | Merkle |  | STOREW | 0 | 2,816 | 
| fibonacci_program | PhantomAir |  | PHANTOM | 0 | 12 | 

| group | air_name | segment | rows | prep_cols | perm_cols | main_cols | cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | AccessAdapterAir<8> | 0 | 64 |  | 24 | 17 | 2,624 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 3 | 8 | 2 | 655,360 | 
| fibonacci_program | MemoryMerkleAir<8> | 0 | 512 |  | 20 | 32 | 26,624 | 
| fibonacci_program | PersistentBoundaryAir<8> | 0 | 64 |  | 12 | 20 | 2,048 | 
| fibonacci_program | PhantomAir | 0 | 2 |  | 12 | 6 | 36 | 
| fibonacci_program | Poseidon2PeripheryAir<BabyBearParameters>, 1> | 0 | 256 |  | 8 | 300 | 78,848 | 
| fibonacci_program | ProgramAir | 0 | 4,096 |  | 8 | 10 | 73,728 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 2 | 8 | 1 | 4,718,592 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 2 | 8 | 1 | 2,359,296 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<4, 8> | 0 | 1,048,576 |  | 80 | 36 | 121,634,816 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<4, 8> | 0 | 524,288 |  | 40 | 37 | 40,370,176 | 
| fibonacci_program | VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<4, 8> | 0 | 2 |  | 52 | 53 | 210 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<4> | 0 | 262,144 |  | 48 | 26 | 19,398,656 | 
| fibonacci_program | VmAirWrapper<Rv32BranchAdapterAir, BranchLessThanCoreAir<4, 8> | 0 | 8 |  | 56 | 32 | 704 | 
| fibonacci_program | VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir> | 0 | 131,072 |  | 44 | 18 | 8,126,464 | 
| fibonacci_program | VmAirWrapper<Rv32HintStoreAdapterAir, Rv32HintStoreCoreAir> | 0 | 4 |  | 36 | 26 | 248 | 
| fibonacci_program | VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir> | 0 | 16 |  | 36 | 28 | 1,024 | 
| fibonacci_program | VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<4> | 0 | 32 |  | 72 | 40 | 3,584 | 
| fibonacci_program | VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir> | 0 | 16 |  | 28 | 21 | 784 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 1 | 12 | 4 | 32 | 

| group | chip_name | segment | rows_used |
| --- | --- | --- | --- |
| fibonacci_program | <Rv32BaseAluAdapterAir,BaseAluCoreAir<4, 8>> | 0 | 900,054 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,LessThanCoreAir<4, 8>> | 0 | 300,002 | 
| fibonacci_program | <Rv32BaseAluAdapterAir,ShiftCoreAir<4, 8>> | 0 | 2 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchEqualCoreAir<4>> | 0 | 200,009 | 
| fibonacci_program | <Rv32BranchAdapterAir,BranchLessThanCoreAir<4, 8>> | 0 | 5 | 
| fibonacci_program | <Rv32CondRdWriteAdapterAir,Rv32JalLuiCoreAir> | 0 | 100,010 | 
| fibonacci_program | <Rv32HintStoreAdapterAir,Rv32HintStoreCoreAir> | 0 | 3 | 
| fibonacci_program | <Rv32JalrAdapterAir,Rv32JalrCoreAir> | 0 | 13 | 
| fibonacci_program | <Rv32LoadStoreAdapterAir,LoadStoreCoreAir<4>> | 0 | 28 | 
| fibonacci_program | <Rv32RdWriteAdapterAir,Rv32AuipcCoreAir> | 0 | 9 | 
| fibonacci_program | AccessAdapter<8> | 0 | 36 | 
| fibonacci_program | Arc<BabyBearParameters>, 1> | 0 | 228 | 
| fibonacci_program | BitwiseOperationLookupAir<8> | 0 | 65,536 | 
| fibonacci_program | Boundary | 0 | 36 | 
| fibonacci_program | Merkle | 0 | 280 | 
| fibonacci_program | PhantomAir | 0 | 2 | 
| fibonacci_program | ProgramChip | 0 | 3,275 | 
| fibonacci_program | RangeTupleCheckerAir<2> | 0 | 524,288 | 
| fibonacci_program | VariableRangeCheckerAir | 0 | 262,144 | 
| fibonacci_program | VmConnectorAir | 0 | 2 | 

| group | cycle_tracker_span | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- | --- |
| fibonacci_program |  |  | ADD | 0 | 2 | 
| fibonacci_program |  |  | AUIPC | 0 | 3 | 
| fibonacci_program |  |  | JALR | 0 | 1 | 
| fibonacci_program |  |  | LOADW | 0 | 1 | 
| fibonacci_program | __start |  | ADD | 0 | 1 | 
| fibonacci_program | __start |  | AUIPC | 0 | 1 | 
| fibonacci_program | __start |  | JALR | 0 | 1 | 
| fibonacci_program | __start |  | STOREW | 0 | 1 | 
| fibonacci_program | __start;main |  | ADD | 0 | 900,019 | 
| fibonacci_program | __start;main |  | AUIPC | 0 | 3 | 
| fibonacci_program | __start;main |  | BEQ | 0 | 100,001 | 
| fibonacci_program | __start;main |  | BNE | 0 | 100,002 | 
| fibonacci_program | __start;main |  | JAL | 0 | 100,001 | 
| fibonacci_program | __start;main |  | JALR | 0 | 4 | 
| fibonacci_program | __start;main |  | LOADW | 0 | 7 | 
| fibonacci_program | __start;main |  | LUI | 0 | 1 | 
| fibonacci_program | __start;main |  | OR | 0 | 1 | 
| fibonacci_program | __start;main |  | SLTU | 0 | 300,000 | 
| fibonacci_program | __start;main |  | STOREW | 0 | 8 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | ADD | 0 | 7 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | AUIPC | 0 | 2 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | BEQ | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | HINT_STOREW | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | JALR | 0 | 3 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | LOADW | 0 | 2 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | LUI | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | PHANTOM | 0 | 2 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E |  | STOREW | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | ADD | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | BGEU | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | JALR | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | LUI | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | SLTU | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | SUB | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;_ZN4core5alloc6layout6Layout19is_size_align_valid17h3e0877a8b80d8b42E |  | XOR | 0 | 2 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | ADD | 0 | 9 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | AND | 0 | 2 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | BLTU | 0 | 2 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | BNE | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | JALR | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LOADW | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | LUI | 0 | 4 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | SLTU | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | STOREW | 0 | 1 | 
| fibonacci_program | __start;main;_ZN6openvm2io4read6Reader3new17h3b34e953a5496fe6E;__rust_alloc_zeroed |  | SUB | 0 | 1 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | ADD | 0 | 6 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | BEQ | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | BGEU | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | BNE | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | HINT_STOREW | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | JALR | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | LOADW | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | LUI | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | SLL | 0 | 2 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | STOREW | 0 | 4 | 
| fibonacci_program | __start;main;_ZN82_$LT$openvm..io..read..Reader$u20$as$u20$openvm..serde..deserializer..WordRead$GT$10read_words17h7c309b7f2dba9782E |  | SUB | 0 | 2 | 

| group | dsl_ir | opcode | segment | frequency |
| --- | --- | --- | --- | --- |
| fibonacci_program |  | ADD | 0 | 900,045 | 
| fibonacci_program |  | AND | 0 | 2 | 
| fibonacci_program |  | AUIPC | 0 | 9 | 
| fibonacci_program |  | BEQ | 0 | 100,004 | 
| fibonacci_program |  | BGEU | 0 | 3 | 
| fibonacci_program |  | BLTU | 0 | 2 | 
| fibonacci_program |  | BNE | 0 | 100,005 | 
| fibonacci_program |  | HINT_STOREW | 0 | 3 | 
| fibonacci_program |  | JAL | 0 | 100,001 | 
| fibonacci_program |  | JALR | 0 | 13 | 
| fibonacci_program |  | LOADW | 0 | 13 | 
| fibonacci_program |  | LUI | 0 | 9 | 
| fibonacci_program |  | OR | 0 | 1 | 
| fibonacci_program |  | PHANTOM | 0 | 2 | 
| fibonacci_program |  | SLL | 0 | 2 | 
| fibonacci_program |  | SLTU | 0 | 300,002 | 
| fibonacci_program |  | STOREW | 0 | 15 | 
| fibonacci_program |  | SUB | 0 | 4 | 
| fibonacci_program |  | XOR | 0 | 2 | 

| group | segment | trace_gen_time_ms | total_proof_time_ms | total_cycles | total_cells | stark_prove_excluding_trace_time_ms | quotient_poly_compute_time_ms | quotient_poly_commit_time_ms | perm_trace_commit_time_ms | pcs_opening_time_ms | main_trace_commit_time_ms | main_cells_used | generate_perm_trace_time_ms | execute_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fibonacci_program | 0 | 251 | 10,987 | 1,500,137 | 197,453,854 | 5,310 | 843 | 474 | 1,611 | 1,407 | 794 | 51,503,940 | 179 | 5,426 | 

</details>


<details>
<summary>Flamegraphs</summary>

[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.air_name.cells_used.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.frequency.reverse.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.frequency.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/fibonacci-65670f9ed753909d30b64615a6deccbf12beedda-fibonacci_program.dsl_ir.opcode.frequency.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.air_name.cells_used.reverse.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.air_name.cells_used.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.frequency.reverse.svg)
[![](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.frequency.svg)](s3://openvm-public-data-sandbox-us-east-1/benchmark/github/flamegraphs/65670f9ed753909d30b64615a6deccbf12beedda/verify_fibair-65670f9ed753909d30b64615a6deccbf12beedda-verify_fibair.dsl_ir.opcode.frequency.svg)

</details>

Commit: https://github.com/openvm-org/openvm/commit/65670f9ed753909d30b64615a6deccbf12beedda

Max Segment Length: 1048476

Instance Type: 64cpu-linux-arm64

Memory Allocator: mimalloc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/12629390168)