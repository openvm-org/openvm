| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 473 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 7,416 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 4,157 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 665 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 200 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 231 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-679192d6017d584bc1a2bf78d6acf058f83987c2.md) | 2,046 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/679192d6017d584bc1a2bf78d6acf058f83987c2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31515919315)
