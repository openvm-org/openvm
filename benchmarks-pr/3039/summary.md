| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 416 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 8,751 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 4,196 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 575 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 281 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-f540f1423ea1025d00ef5191d95df64a53e2b0dc.md) | 1,916 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f540f1423ea1025d00ef5191d95df64a53e2b0dc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29649559537)
