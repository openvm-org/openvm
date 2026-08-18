#!/usr/bin/env bash

readonly OPENVM_APT_OPTIONS=(
    -o Acquire::Retries=3
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
    -o DPkg::Lock::Timeout=120
)

use_ubuntu_archive_fallback() {
    local source_file
    local changed=false

    while IFS= read -r -d '' source_file; do
        if sudo grep -Eq 'https?://([[:alnum:]-]+\.)*ec2\.archive\.ubuntu\.com/ubuntu/?' "$source_file"; then
            sudo sed -Ei \
                's#https?://([[:alnum:]-]+\.)*ec2\.archive\.ubuntu\.com/ubuntu/?#https://archive.ubuntu.com/ubuntu/#g' \
                "$source_file"
            changed=true
        fi
    done < <(
        sudo find /etc/apt -maxdepth 2 -type f \
            \( -name sources.list -o -name '*.list' -o -name '*.sources' \) -print0
    )

    [[ "$changed" == true ]]
}

apt_get() {
    if use_ubuntu_archive_fallback; then
        echo "Using archive.ubuntu.com instead of the regional Ubuntu archive." >&2
    fi
    sudo apt-get "${OPENVM_APT_OPTIONS[@]}" "$@"
}
