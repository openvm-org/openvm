| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 1,811 |  4,000,051 |  428 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 13,779 |  14,365,133 |  2,337 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 8,209 |  11,167,961 |  903 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 1,535 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 599 |  112,210 |  263 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 742 |  592,827 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-7ffd71690fd28d1026e852f76085e184fe964d60.md) | 1,897 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7ffd71690fd28d1026e852f76085e184fe964d60

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25926653855)
