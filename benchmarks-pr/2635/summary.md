| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/fibonacci-61ee02051eb604c485ac49e948a254e80d67ce0b.md) | 3,849 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/keccak-61ee02051eb604c485ac49e948a254e80d67ce0b.md) | 15,752 |  1,235,218 |  2,210 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/regex-61ee02051eb604c485ac49e948a254e80d67ce0b.md) | 1,426 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/ecrecover-61ee02051eb604c485ac49e948a254e80d67ce0b.md) | 633 |  122,348 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/pairing-61ee02051eb604c485ac49e948a254e80d67ce0b.md) | 916 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/kitchen_sink-61ee02051eb604c485ac49e948a254e80d67ce0b.md) | 2,388 |  154,763 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/61ee02051eb604c485ac49e948a254e80d67ce0b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23817392395)
