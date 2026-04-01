| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-1ad437233773728d01a10959bea6d39cf0e0423f.md) | 3,820 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-1ad437233773728d01a10959bea6d39cf0e0423f.md) | 18,600 |  18,655,329 |  3,321 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-1ad437233773728d01a10959bea6d39cf0e0423f.md) | 1,430 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-1ad437233773728d01a10959bea6d39cf0e0423f.md) | 651 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-1ad437233773728d01a10959bea6d39cf0e0423f.md) | 902 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-1ad437233773728d01a10959bea6d39cf0e0423f.md) | 2,276 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1ad437233773728d01a10959bea6d39cf0e0423f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23866532317)
