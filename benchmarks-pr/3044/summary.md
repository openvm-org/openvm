| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/fibonacci-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 405 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/keccak-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 8,596 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/sha2_bench-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 4,180 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/regex-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 568 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/ecrecover-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 219 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/pairing-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 283 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3044/kitchen_sink-49bf87f76ff830dbd83418380a0a0569ef08fd4e.md) | 1,934 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/49bf87f76ff830dbd83418380a0a0569ef08fd4e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29657920281)
