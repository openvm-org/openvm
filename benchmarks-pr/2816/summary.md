| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/fibonacci-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 1,900 |  4,000,051 |  522 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/keccak-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 13,304 |  14,365,133 |  2,166 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/sha2_bench-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 9,408 |  11,167,961 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/regex-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 1,580 |  4,090,656 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/ecrecover-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 607 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/pairing-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 742 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/kitchen_sink-555ad13649442dfe35e909367bc0cdd146be08a0.md) | 1,885 |  1,979,971 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/555ad13649442dfe35e909367bc0cdd146be08a0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26474626574)
