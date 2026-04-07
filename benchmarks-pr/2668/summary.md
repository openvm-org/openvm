| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/fibonacci-0473bff96d67258421ae6265284927e5b87902d6.md) | 3,806 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/keccak-0473bff96d67258421ae6265284927e5b87902d6.md) | 16,090 |  1,235,218 |  2,253 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/regex-0473bff96d67258421ae6265284927e5b87902d6.md) | 1,428 |  4,136,694 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/ecrecover-0473bff96d67258421ae6265284927e5b87902d6.md) | 636 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/pairing-0473bff96d67258421ae6265284927e5b87902d6.md) | 921 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/kitchen_sink-0473bff96d67258421ae6265284927e5b87902d6.md) | 2,417 |  154,763 |  418 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/fibonacci_e2e-0473bff96d67258421ae6265284927e5b87902d6.md) | 1,723 |  12,000,265 |  458 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/regex_e2e-0473bff96d67258421ae6265284927e5b87902d6.md) | 870 |  4,136,694 |  189 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/ecrecover_e2e-0473bff96d67258421ae6265284927e5b87902d6.md) | 400 |  122,348 |  150 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/pairing_e2e-0473bff96d67258421ae6265284927e5b87902d6.md) | 539 |  1,745,757 |  155 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/kitchen_sink_e2e-0473bff96d67258421ae6265284927e5b87902d6.md) | 2,376 |  154,763 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0473bff96d67258421ae6265284927e5b87902d6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24105684918)
