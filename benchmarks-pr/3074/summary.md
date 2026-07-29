| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 479 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 7,343 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 4,097 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 652 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 232 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 239 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21.md) | 2,036 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2799cd1fe6c12dd8e4a3fbae5ba03d2c9d5cea21

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30500695322)
