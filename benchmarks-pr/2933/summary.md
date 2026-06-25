| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/fibonacci-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 1,040 |  4,000,051 |  397 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/keccak-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 16,204 |  14,365,133 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/sha2_bench-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 8,293 |  11,167,961 |  1,006 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/regex-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 1,202 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/ecrecover-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 432 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/pairing-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 600 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2933/kitchen_sink-b584d308cd3c3498bf5e14a36257b679c890d5d9.md) | 3,868 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b584d308cd3c3498bf5e14a36257b679c890d5d9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28198617376)
