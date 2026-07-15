#!/usr/bin/env bash
#
# Ensures a working NVIDIA driver, switching to the open kernel module flavor
# if needed. Blackwell GPUs require open kernel modules.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    exit 0
fi

KERNEL_RELEASE="$(uname -r)"

installed_nvidia_packages() {
    dpkg-query -W -f='${binary:Package}\n' 2>/dev/null || true
}

detect_driver_series() {
    local package series

    if [[ -n "${NVIDIA_DRIVER_SERIES:-}" ]]; then
        echo "$NVIDIA_DRIVER_SERIES"
        return
    fi

    while IFS= read -r package; do
        if [[ "$package" == linux-modules-nvidia-*-open-"$KERNEL_RELEASE" ||
              "$package" == linux-modules-nvidia-*-"$KERNEL_RELEASE" ]]; then
            series="${package#linux-modules-nvidia-}"
            series="${series%-"$KERNEL_RELEASE"}"
            series="${series%-open}"
            if [[ "$series" =~ ^[0-9]+$ ]]; then
                echo "$series"
                return
            fi
        fi
    done < <(installed_nvidia_packages)

    installed_nvidia_packages |
        sed -nE 's/^(nvidia-utils|nvidia-driver|nvidia-dkms|linux-modules-nvidia)-([0-9]+)(-.+)?$/\2/p' |
        sort -V |
        tail -n1
}

DRIVER_SERIES="$(detect_driver_series)"
if ! [[ "$DRIVER_SERIES" =~ ^[0-9]+$ ]]; then
    echo "Unable to detect installed NVIDIA driver series; set NVIDIA_DRIVER_SERIES to a numeric series." >&2
    exit 1
fi

OPEN_MODULE_PKG="linux-modules-nvidia-${DRIVER_SERIES}-open-${KERNEL_RELEASE}"

remove_closed_kernel_modules() {
    mapfile -t closed_pkgs < <(
        installed_nvidia_packages |
            awk -v series="$DRIVER_SERIES" '
                $0 ~ "^(nvidia-driver|nvidia-dkms|nvidia-kernel-source)-" series "($|-)" && $0 !~ "-open($|-)" {
                    print
                    next
                }
                $0 ~ "^linux-modules-nvidia-" series "($|-)" && $0 !~ "-open($|-)" {
                    print
                }
            '
    )

    if ((${#closed_pkgs[@]})); then
        sudo apt-get remove -y "${closed_pkgs[@]}"
    fi
}

has_apt_candidate() {
    local package="$1"
    local candidate
    candidate="$(apt-cache policy "$package" | awk '/Candidate:/ { print $2; exit }')"
    [[ -n "$candidate" && "$candidate" != "(none)" ]]
}

echo "nvidia-smi not functional; switching to open kernel module driver flavor..."
set -ex
sudo apt-get update -qq
remove_closed_kernel_modules

# Prefer the prebuilt open modules for the running kernel. Installing the
# nvidia-driver-*-open meta package here pulls DKMS, which conflicts with the
# same .ko files shipped by the prebuilt linux-modules package.
if has_apt_candidate "$OPEN_MODULE_PKG"; then
    sudo apt-get install -y --no-install-recommends "$OPEN_MODULE_PKG"
else
    sudo apt-get install -y --no-install-recommends \
        "nvidia-driver-${DRIVER_SERIES}-open" \
        "nvidia-dkms-${DRIVER_SERIES}-open" \
        "linux-headers-${KERNEL_RELEASE}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    sudo apt-get install -y --no-install-recommends "nvidia-utils-${DRIVER_SERIES}"
fi

sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
sudo modprobe nvidia
sudo modprobe nvidia_uvm
nvidia-smi
