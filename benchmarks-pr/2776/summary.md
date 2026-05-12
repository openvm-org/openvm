| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 3,827 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 18,564 |  18,655,329 |  3,325 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 9,083 |  14,793,960 |  1,418 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 1,405 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 638 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 893 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-2b7a464f0e232ab315006930148fb8b32c7178be.md) | 2,082 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2b7a464f0e232ab315006930148fb8b32c7178be

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25744805315)
