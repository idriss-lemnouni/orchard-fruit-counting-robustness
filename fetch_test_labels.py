"""Fetch just the official COCO annotation member from the MinneApple test ZIP."""
import argparse
import hashlib
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from prepare import COCO_MEMBER, TEST_BYTES

TEST_URL = ("https://conservancy.umn.edu/server/api/core/bitstreams/"
            "7856047c-1f23-43ef-a50d-9327ec2ae0ff/content")


class ArchiveRangeReader:
    def __init__(self):
        self.position = 0

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        position = (0, self.position, TEST_BYTES)[whence] + offset
        if position < 0:
            raise OSError("Seek before start of original ZIP")
        self.position = position
        return position

    def read(self, length=-1):
        start = self.position
        end = TEST_BYTES if length < 0 else min(start + length, TEST_BYTES)
        if start >= end:
            return b""
        if end - start > 12_000_000:
            raise ValueError("Requested ZIP range is unexpectedly large")
        headers = {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}
        for attempt in range(3):
            try:
                with urlopen(Request(TEST_URL, headers=headers), timeout=55) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{TEST_BYTES}":
                        raise ValueError("The university server did not return the requested ZIP byte range")
                    content = response.read(end - start + 1)
                    if len(content) != end - start:
                        raise OSError("Truncated ZIP range")
                    self.position = end
                    return content
            except (URLError, TimeoutError, HTTPException, OSError):
                if attempt == 2:
                    raise


def write_file(path, data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def fetch(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(ArchiveRangeReader()) as archive:
        annotation = archive.read(COCO_MEMBER)
    write_file(output, annotation)
    print(f"COCO member CRC verified. SHA-256 {hashlib.sha256(annotation).hexdigest()}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=Path("data/raw/ground_truth.json"))
    args = cli.parse_args()
    fetch(args.out)
