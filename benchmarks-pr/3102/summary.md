| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 466 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 7,414 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 4,127 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 648 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 226 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 235 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-016fcc86748deb78634ce865f77e3c6937cc3588.md) | 2,044 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/016fcc86748deb78634ce865f77e3c6937cc3588

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31039502738)
