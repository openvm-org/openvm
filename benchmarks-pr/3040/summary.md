| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 418 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 8,567 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 4,255 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 579 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 219 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 280 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec.md) | 1,917 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cb7dedc71742ad5bb84ac1d41cae6bf0de2e93ec

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29656128530)
