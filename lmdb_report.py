#!/usr/bin/env python3
"""
lmdb_report.py

Utility to report on and optionally warm up an LMDB cache.

Usage:
    python lmdb_report.py --lmdb-dir /path/to/lmdb [--warm]

Options:
    --lmdb-dir PATH   Path to the LMDB directory.
    --warm            If provided, will scan all keys first to fault pages into memory.
"""
import argparse
import lmdb


def lmdb_usage_report(lmdb_dir: str, warm: bool = False):
    """
    Prints number of entries, page size, allocated pages, map_size,
    and percentage of map used by the current LMDB environment.
    If warm=True, does a full scan of all keys first (to fault in all pages).
    """
    env = lmdb.open(
        lmdb_dir,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    if warm:
        print("Warming LMDB cache by scanning all keys…")
        with env.begin() as txn:
            cursor = txn.cursor()
            for _ in cursor:
                pass
        print("Warm-up complete.\n")

    stat = env.stat()
    info = env.info()

    psize       = stat['psize']       # bytes per page
    entries     = stat['entries']     # total key/value pairs
    last_pgno   = info['last_pgno']   # highest page number allocated
    map_size    = info['map_size']    # total bytes reserved on disk

    used_bytes  = last_pgno * psize
    used_pct    = used_bytes / map_size * 100.0

    print(f"LMDB stats for: {lmdb_dir}")
    print(f"  entries     : {entries}")
    print(f"  page size   : {psize/1024:.1f} KiB")
    print(f"  map size    : {map_size/1024**2:.1f} MiB")
    print(f"  pages alloc.: {last_pgno}")
    print(f"  used bytes  : {used_bytes/1024**2:.1f} MiB ({used_pct:.1f}% of map)")


def main():
    parser = argparse.ArgumentParser(description="LMDB usage reporter")
    parser.add_argument(
        "--lmdb-dir", type=str, required=True,
        help="Path to the LMDB directory"
    )
    parser.add_argument(
        "--warm", action="store_true",
        help="Scan all keys to warm the OS cache before reporting"
    )
    args = parser.parse_args()

    lmdb_usage_report(args.lmdb_dir, warm=args.warm)


if __name__ == "__main__":
    main()