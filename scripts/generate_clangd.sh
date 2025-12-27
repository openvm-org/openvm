#! /bin/bash

# IMPORTANT: Run this from the OpenVM workspace root

cargo run -p openvm-scripts --features cuda --bin generate-clangd
