| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 1,031 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 16,326 |  14,365,133 |  3,015 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 8,204 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 1,212 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 442 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 597 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f.md) | 3,938 |  1,979,971 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b5d2f9df1af1ee40159e371a9ce89d9f6546ed3f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27843000822)
