| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/fibonacci-7fd4610fa7831c691de3403135fcf1ab95b80d0f.md) | 3,847 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/keccak-7fd4610fa7831c691de3403135fcf1ab95b80d0f.md) | 15,687 |  1,235,218 |  2,200 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/regex-7fd4610fa7831c691de3403135fcf1ab95b80d0f.md) | 1,415 |  4,136,694 |  367 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/ecrecover-7fd4610fa7831c691de3403135fcf1ab95b80d0f.md) | 643 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/pairing-7fd4610fa7831c691de3403135fcf1ab95b80d0f.md) | 920 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/kitchen_sink-7fd4610fa7831c691de3403135fcf1ab95b80d0f.md) | 2,366 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7fd4610fa7831c691de3403135fcf1ab95b80d0f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23816945585)
