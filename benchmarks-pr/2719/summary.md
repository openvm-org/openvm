| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 1,860 |  4,000,051 |  534 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 13,414 |  14,365,133 |  2,215 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 9,489 |  11,167,961 |  1,282 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 1,588 |  4,090,656 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 642 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 755 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-c5b9459ca4370311385df6bd16588dedf14898d1.md) | 2,091 |  1,979,971 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c5b9459ca4370311385df6bd16588dedf14898d1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25173132488)
