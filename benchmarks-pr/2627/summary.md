| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-0c998cc1db73a3c7774c46a4f1af772a3adc3dae.md) | 3,870 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-0c998cc1db73a3c7774c46a4f1af772a3adc3dae.md) | 15,560 |  1,235,218 |  2,180 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-0c998cc1db73a3c7774c46a4f1af772a3adc3dae.md) | 1,417 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-0c998cc1db73a3c7774c46a4f1af772a3adc3dae.md) | 632 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-0c998cc1db73a3c7774c46a4f1af772a3adc3dae.md) | 919 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-0c998cc1db73a3c7774c46a4f1af772a3adc3dae.md) | 2,363 |  154,763 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c998cc1db73a3c7774c46a4f1af772a3adc3dae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23868848531)
