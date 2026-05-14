| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 1,894 |  4,000,051 |  542 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 13,396 |  14,365,133 |  2,204 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 9,476 |  11,167,961 |  1,416 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 1,593 |  4,090,656 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 633 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 767 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-c09775a2815d84e5ffe9ccd68134d44d7702f569.md) | 2,035 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c09775a2815d84e5ffe9ccd68134d44d7702f569

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25855168272)
