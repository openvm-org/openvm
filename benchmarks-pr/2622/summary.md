| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-da62e677245c0889a21f326470cc2cfd9357a97b.md) | 3,880 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-da62e677245c0889a21f326470cc2cfd9357a97b.md) | 15,657 |  1,235,218 |  2,176 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-da62e677245c0889a21f326470cc2cfd9357a97b.md) | 1,416 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-da62e677245c0889a21f326470cc2cfd9357a97b.md) | 636 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-da62e677245c0889a21f326470cc2cfd9357a97b.md) | 919 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-da62e677245c0889a21f326470cc2cfd9357a97b.md) | 2,393 |  154,763 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/da62e677245c0889a21f326470cc2cfd9357a97b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23607033197)
