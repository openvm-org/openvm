| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/fibonacci-4710d91acc638fbfa102e334aaf93c68fcbd685e.md) | 3,857 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/keccak-4710d91acc638fbfa102e334aaf93c68fcbd685e.md) | 15,550 |  1,235,218 |  2,182 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/regex-4710d91acc638fbfa102e334aaf93c68fcbd685e.md) | 1,434 |  4,136,694 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/ecrecover-4710d91acc638fbfa102e334aaf93c68fcbd685e.md) | 640 |  122,348 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/pairing-4710d91acc638fbfa102e334aaf93c68fcbd685e.md) | 928 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2642/kitchen_sink-4710d91acc638fbfa102e334aaf93c68fcbd685e.md) | 2,381 |  154,763 |  418 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4710d91acc638fbfa102e334aaf93c68fcbd685e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23848786029)
