| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 892 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 8,677 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 4,258 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 736 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 501 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 478 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4.md) | 2,348 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7d34e2dcffaa4b7c1a38f70bcaf0cc8d56bab6f4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31050142720)
