| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/fibonacci-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 3,074 |  12,000,265 |  681 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/keccak-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 16,280 |  18,655,329 |  3,018 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/sha2_bench-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 9,125 |  14,793,960 |  1,113 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/regex-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 1,173 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/ecrecover-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 600 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/pairing-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 940 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/kitchen_sink-eb65141cc7e4b0cc39d9622b573294775fd460c5.md) | 4,074 |  2,579,903 |  873 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eb65141cc7e4b0cc39d9622b573294775fd460c5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27850791579)
