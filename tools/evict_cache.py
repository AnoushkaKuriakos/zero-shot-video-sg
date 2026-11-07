#!/usr/bin/env python3
"""Evict our files from the Linux page cache without root (posix_fadvise DONTNEED).

Why: on this WSL box the page cache (model weights, dataset frames) keeps ~55 GB of
Windows commit pinned in vmmemWSL; Windows has no pagefile, so free commit falls under
the guard.sh limit as soon as a 7B model loads. We are not in sudoers, so
`echo 3 > /proc/sys/vm/drop_caches` is not an option; fadvise works on any file we can
read and only drops clean pages (nothing is lost). WSL's free page reporting then hands
the freed RAM back to Windows within a few seconds.

    python tools/evict_cache.py [dir ...]     # default: weights, data, envs, outputs
"""
import os
import sys

HOME = os.path.expanduser("~")
DEFAULT_DIRS = [f"{HOME}/.cache", os.getcwd(), f"{HOME}/miniconda3"]


def cached_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return 0


def evict(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


def main():
    dirs = sys.argv[1:] or DEFAULT_DIRS
    before = cached_kb()
    for d in dirs:
        for root, _, files in os.walk(d):
            for name in files:
                p = os.path.join(root, name)
                if os.path.isfile(p) and not os.path.islink(p):
                    evict(p)
    after = cached_kb()
    print(f"page cache {before / 2**20:.1f}G -> {after / 2**20:.1f}G")


if __name__ == "__main__":
    main()
