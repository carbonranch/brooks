"""Read single members out of a large remote zip (Zenodo) with HTTP range requests, without downloading the archive.

zipfile.ZipFile only needs a seekable file object; this one fetches byte ranges on demand (with a read-ahead block) and
caches them, so listing the central directory costs a few MB and extracting a member costs about its own size.

    z = RemoteZip("https://zenodo.org/records/18554693/files/Test.zip?download=1")
    names = z.namelist(); data = z.read(names[0])
"""
from __future__ import annotations

import io
import zipfile

import requests

BLOCK = 4 << 20       # 4 MB read-ahead


class _RangeFile(io.RawIOBase):
    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url; self.s = session or requests.Session(); self.pos = 0; self.cache: dict[int, bytes] = {}
        h = self.s.head(url, allow_redirects=True, timeout=60); h.raise_for_status()
        self.size = int(h.headers["Content-Length"]); self.url = h.url
        if h.headers.get("Accept-Ranges", "bytes") != "bytes":
            raise RuntimeError("server does not accept range requests")

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]; return self.pos

    def _block(self, k: int) -> bytes:
        if k not in self.cache:
            a = k * BLOCK; b = min(self.size, a + BLOCK) - 1
            r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120); r.raise_for_status()
            self.cache[k] = r.content
            if len(self.cache) > 64:                       # keep memory bounded: drop the oldest block
                self.cache.pop(next(iter(self.cache)))
        return self.cache[k]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray(); end = min(self.size, self.pos + n)
        while self.pos < end:
            k, off = divmod(self.pos, BLOCK); blk = self._block(k)
            take = blk[off:off + (end - self.pos)]; out += take; self.pos += len(take)
            if not take:
                break
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


class RemoteZip(zipfile.ZipFile):
    def __init__(self, url: str):
        self.rf = _RangeFile(url)
        super().__init__(io.BufferedReader(self.rf, buffer_size=BLOCK))

    @property
    def total_bytes(self) -> int:
        return self.rf.size
