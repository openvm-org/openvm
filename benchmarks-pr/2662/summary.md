| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-0d0d427e98165f862291f60350ba32daf1254078.md) | 3,840 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-0d0d427e98165f862291f60350ba32daf1254078.md) | 18,590 |  18,655,329 |  3,329 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-0d0d427e98165f862291f60350ba32daf1254078.md) | 1,420 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-0d0d427e98165f862291f60350ba32daf1254078.md) | 647 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-0d0d427e98165f862291f60350ba32daf1254078.md) | 901 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-0d0d427e98165f862291f60350ba32daf1254078.md) | 2,145 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0d0d427e98165f862291f60350ba32daf1254078

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24199123839)
