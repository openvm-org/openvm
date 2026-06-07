| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 1,549 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 13,456 |  14,365,133 |  2,355 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 8,893 |  11,167,961 |  1,399 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 1,562 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 484 |  112,210 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 597 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 3,805 |  1,979,971 |  950 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci_e2e-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 815 |  4,000,051 |  196 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex_e2e-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 910 |  4,090,656 |  173 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover_e2e-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 333 |  112,210 |  134 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing_e2e-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 397 |  592,827 |  128 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink_e2e-e582d737ece2c6d8459e952b5ea3774b3986747e.md) | 2,063 |  1,979,971 |  393 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e582d737ece2c6d8459e952b5ea3774b3986747e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27091294193)
