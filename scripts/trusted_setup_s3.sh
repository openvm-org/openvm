#!/bin/bash

if [[ -z $1 ]]; then
    maxk=24
else
    maxk=$1
fi
echo "maxk=$maxk"

PARAMS_DIR="$HOME/.openvm/params/"
mkdir -p $PARAMS_DIR
for k in $(seq 10 $maxk)
do
    pkey_file="kzg_bn254_${k}.srs"
    if test -f $pkey_file; then
        echo "$pkey_file already exists"
    else
        echo "downloading $pkey_file"
        s5cmd --no-sign-request cp --concurrency 10 "s3://axiom-crypto/challenge_0085/${pkey_file}" $PARAMS_DIR
    fi
done
