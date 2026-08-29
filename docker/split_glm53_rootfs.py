#!/usr/bin/python3
"""Move large runtime trees into deterministic, size-bounded image layers."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import NamedTuple


ATOM_LIMIT = 512 * 1024 * 1024
BUCKET_COUNT = 32
BUCKET_LIMIT = 1_250_000_000
LAYER_ROOT = Path("/__image_layers")
SOURCES = (
    Path("/opt"),
    Path("/root"),
    Path("/sgl-workspace"),
    Path("/usr/include"),
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/libexec"),
    Path("/usr/local"),
    Path("/usr/share"),
)


class Node(NamedTuple):
    path: Path
    size: int
    children: tuple["Node", ...]


def scan_tree(path: Path, directory_stats: dict[Path, os.stat_result]) -> Node:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        return Node(path, metadata.st_size, ())

    directory_stats[path] = metadata
    children = tuple(
        scan_tree(Path(entry.path), directory_stats) for entry in sorted(os.scandir(path), key=lambda item: item.name)
    )
    return Node(path, metadata.st_size + sum(child.size for child in children), children)


def iter_atoms(node: Node):
    if not node.children or node.size <= ATOM_LIMIT:
        yield node.path, node.size
        return
    for child in node.children:
        yield from iter_atoms(child)


def make_parent_directories(
    source: Path,
    destination: Path,
    directory_stats: dict[Path, os.stat_result],
    created_directories: dict[Path, Path],
) -> None:
    source_parent = Path("/")
    destination_parent = destination.parent
    relative_parents = source.relative_to("/").parts[:-1]
    destination_cursor = LAYER_ROOT / destination.relative_to(LAYER_ROOT).parts[0]

    for component in relative_parents:
        source_parent /= component
        destination_cursor /= component
        if not destination_cursor.exists():
            destination_cursor.mkdir()
            created_directories[destination_cursor] = source_parent

    assert destination_cursor == destination_parent


def restore_directory_metadata(destination: Path, source: Path, directory_stats: dict[Path, os.stat_result]) -> None:
    metadata = directory_stats[source]
    os.chown(destination, metadata.st_uid, metadata.st_gid, follow_symlinks=False)
    os.chmod(destination, stat.S_IMODE(metadata.st_mode), follow_symlinks=False)
    os.utime(
        destination,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        follow_symlinks=False,
    )


def main() -> None:
    Path(__file__).unlink()
    LAYER_ROOT.mkdir(mode=0o755)

    directory_stats: dict[Path, os.stat_result] = {}
    for source in SOURCES:
        parent = source.parent
        while parent != Path("/"):
            directory_stats.setdefault(parent, parent.lstat())
            parent = parent.parent
    nodes = tuple(scan_tree(source, directory_stats) for source in SOURCES)
    atoms = sorted(
        (atom for node in nodes for atom in iter_atoms(node)),
        key=lambda item: (-item[1], str(item[0])),
    )

    bucket_sizes = [0] * BUCKET_COUNT
    assignments: list[tuple[Path, int]] = []
    for source, size in atoms:
        bucket = min(range(BUCKET_COUNT), key=lambda index: (bucket_sizes[index], index))
        bucket_sizes[bucket] += size
        assignments.append((source, bucket))

    largest_bucket = max(bucket_sizes)
    if largest_bucket > BUCKET_LIMIT:
        raise RuntimeError(f"largest rootfs bucket is {largest_bucket} bytes, over {BUCKET_LIMIT}")

    created_directories: dict[Path, Path] = {}
    for index in range(BUCKET_COUNT):
        (LAYER_ROOT / f"{index:02d}").mkdir(mode=0o755)

    for source, bucket in sorted(assignments, key=lambda item: str(item[0])):
        destination = LAYER_ROOT / f"{bucket:02d}" / source.relative_to("/")
        make_parent_directories(source, destination, directory_stats, created_directories)
        source.rename(destination)

    for destination, source in sorted(created_directories.items(), key=lambda item: len(item[0].parts), reverse=True):
        restore_directory_metadata(destination, source, directory_stats)

    for index, size in enumerate(bucket_sizes):
        print(f"rootfs layer {index:02d}: {size} uncompressed bytes")


if __name__ == "__main__":
    main()
