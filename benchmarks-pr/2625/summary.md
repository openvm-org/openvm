| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3.md) | 3,863 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3.md) | 15,777 |  1,235,218 |  2,176 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3.md) | 1,410 |  4,136,694 |  367 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3.md) | 634 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3.md) | 923 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3.md) | 2,393 |  154,763 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e5173cfb9e710de62fe0ceea2e5eb7c10a07c2f3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23646960803)
