| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/fibonacci-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 408 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/keccak-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 8,462 |  14,365,133 |  1,542 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/sha2_bench-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 3,928 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/regex-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 566 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/ecrecover-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 221 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/pairing-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 268 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/kitchen_sink-47957d6dfb8e7c107ea340db122e5d2ab6adf100.md) | 1,905 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/47957d6dfb8e7c107ea340db122e5d2ab6adf100

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29453512892)
