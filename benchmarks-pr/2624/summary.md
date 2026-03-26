| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/fibonacci-2316f2561cb3d2295c15df6b172a7554397216da.md) | 3,794 |  12,000,265 |  932 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/keccak-2316f2561cb3d2295c15df6b172a7554397216da.md) | 15,644 |  1,235,218 |  2,177 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/regex-2316f2561cb3d2295c15df6b172a7554397216da.md) | 1,417 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/ecrecover-2316f2561cb3d2295c15df6b172a7554397216da.md) | 640 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/pairing-2316f2561cb3d2295c15df6b172a7554397216da.md) | 916 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/kitchen_sink-2316f2561cb3d2295c15df6b172a7554397216da.md) | 2,375 |  154,763 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2316f2561cb3d2295c15df6b172a7554397216da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23610958440)
