| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4.md) | 3,805 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4.md) | 15,600 |  1,235,218 |  2,154 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4.md) | 1,450 |  4,136,694 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4.md) | 640 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4.md) | 922 |  1,745,757 |  274 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4.md) | 2,372 |  154,763 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c1f76ca547b033deee6f56d0b0c1e77ea3bd2e4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23605904801)
