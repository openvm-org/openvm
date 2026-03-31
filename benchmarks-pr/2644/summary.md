| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2644/fibonacci-df6abc9dafa300daec6d88d50ac80569f5f552d3.md) | 3,822 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2644/keccak-df6abc9dafa300daec6d88d50ac80569f5f552d3.md) | 15,939 |  1,235,218 |  2,233 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2644/regex-df6abc9dafa300daec6d88d50ac80569f5f552d3.md) | 1,422 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2644/ecrecover-df6abc9dafa300daec6d88d50ac80569f5f552d3.md) | 634 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2644/pairing-df6abc9dafa300daec6d88d50ac80569f5f552d3.md) | 922 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2644/kitchen_sink-df6abc9dafa300daec6d88d50ac80569f5f552d3.md) | 2,383 |  154,763 |  418 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/df6abc9dafa300daec6d88d50ac80569f5f552d3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23823342381)
