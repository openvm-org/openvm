| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 1,560 |  4,000,051 |  435 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 14,026 |  14,365,133 |  2,370 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 9,160 |  11,167,961 |  1,403 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 1,598 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 484 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 607 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-1ef15de87cad201df013dacfe95250c6fb7c14ef.md) | 2,170 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1ef15de87cad201df013dacfe95250c6fb7c14ef

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26879453370)
