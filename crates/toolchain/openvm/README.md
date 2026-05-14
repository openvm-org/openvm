# OpenVM Standard Library

The OpenVM standard library for use in Rust guest programs.

## Regenerating `memcpy.s` / `memset.s`

The two checked-in assembly files in `src/` are produced from musl-libc sources by `scripts/generate_libc_intrinsics.sh` (at the repo root). To refresh them — e.g. when bumping the musl commit or the clang version — run:

```bash
scripts/generate_libc_intrinsics.sh
```

The script accepts optional `--musl-ref <tag|branch|sha>` and `--clang <path>` flags; see its header comment for details.
