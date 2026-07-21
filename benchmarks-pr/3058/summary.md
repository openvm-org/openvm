| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 472 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 7,317 |  14,365,133 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 4,755 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 664 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 226 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 330 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac.md) | 2,608 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dee65ffb6b9bcdc830ea3f96e45c413ad3b3ceac

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29875924004)
