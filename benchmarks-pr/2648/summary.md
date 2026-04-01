| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/fibonacci-cee086ac9270babd2cb1bd731e4e385806123a97.md) | 3,863 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/keccak-cee086ac9270babd2cb1bd731e4e385806123a97.md) | 15,615 |  1,235,218 |  2,181 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/regex-cee086ac9270babd2cb1bd731e4e385806123a97.md) | 1,424 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/ecrecover-cee086ac9270babd2cb1bd731e4e385806123a97.md) | 641 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/pairing-cee086ac9270babd2cb1bd731e4e385806123a97.md) | 921 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/kitchen_sink-cee086ac9270babd2cb1bd731e4e385806123a97.md) | 2,383 |  154,763 |  418 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cee086ac9270babd2cb1bd731e4e385806123a97

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23865730235)
