| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-0cffa2ee454e6050ea04e094e3956dc8109426ae.md) | 3,830 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-0cffa2ee454e6050ea04e094e3956dc8109426ae.md) | 18,744 |  18,655,329 |  3,348 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-0cffa2ee454e6050ea04e094e3956dc8109426ae.md) | 1,438 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-0cffa2ee454e6050ea04e094e3956dc8109426ae.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-0cffa2ee454e6050ea04e094e3956dc8109426ae.md) | 904 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-0cffa2ee454e6050ea04e094e3956dc8109426ae.md) | 2,280 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0cffa2ee454e6050ea04e094e3956dc8109426ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23862504254)
