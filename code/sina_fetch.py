"""Pull every species page from Singing Insects of North America (SINA) and
extract the song paragraph: pulse/chirp rates with the temperature they were
measured at, dominant/carrier frequency, and the song pattern. Writes one row
per species to sina_songs.csv for matching against the month's streaks.

Usage:
    python analysis/callers/sina_fetch.py analysis/callers/runs/month/sina
"""
from __future__ import annotations

import html
import os
import re
import sys
import time
import urllib.request

import pandas as pd

BASE = "https://orthsoc.org/sina/"
LISTS = {"cricket": "cricklist.htm", "katydid": "katylist.htm"}
UA = {"User-Agent": "Mozilla/5.0 (research; caller-inventory matching)"}


def get(url: str, cache_dir: str) -> str:
    fn = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fn):
        return open(fn, encoding="latin-1").read()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        txt = r.read().decode("latin-1", errors="replace")
    open(fn, "w", encoding="latin-1").write(txt)
    time.sleep(0.3)
    return txt


def strip(s: str) -> str:
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>|</p>|</tr>|</h\d>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"[ \t\r\f\v]+", " ", s)


def species_links(list_html: str):
    out = []
    subfam = ""
    # subfamily/family headings are <h2 class="speciestitle" id="..."> ; species
    # links sit inside <h3 class="species"> as <a href="NNNa.htm"><i>name</i></a> - common
    for m in re.finditer(r'<h2 class="speciestitle" id="(\w+)">|<a href="(\d{3}a\.htm)"[^>]*>(.*?)</a>\s*-?\s*([^<\n]*)',
                         list_html, flags=re.S | re.I):
        if m.group(1):
            subfam = m.group(1)
        elif m.group(2):
            out.append((subfam, m.group(2), strip(m.group(3)).strip() + " | " + strip(m.group(4)).strip().rstrip("*").strip()))
    return out


NUM = r"(\d+(?:\.\d+)?)"


def parse_song(text: str) -> dict:
    t = text.replace("\n", " ")
    d = {}
    # the song section: from "Song" heading-ish to the next heading-ish word
    m = re.search(r"Song\b(.*?)(?:Similar species|Identification|Habitat|Season|Remarks|Distribution|Life cycle|References|More information|$)", t, flags=re.S | re.I)
    song = m.group(1).strip() if m else ""
    d["song_text"] = song[:1200]
    # pulse or chirp rates with optional temperature
    # prefer the "Song at 25°C:" paragraph when present; its rates are at 25 degC
    m25 = re.search(r"Song at 25\s*°?\s*C\s*:?(.*?)(?:Song data|Specimen data|Nomenclature|Similar species|Identification|Habitat|Remarks|References|$)", t, flags=re.S | re.I)
    at25 = m25.group(1) if m25 else ""
    d["song_at_25"] = at25.strip()[:600]
    src = at25 if at25 else song
    # "58 p/s", "10 per second", "ca. 2-3 ch/s", "pulse rate of 52/sec", "52 pulses per second"
    rates = re.findall(r"(?<![\d.])" + NUM + r"(?:\s*-\s*" + NUM + r")?\s*(?:p/s|p/sec|pulses?\s*(?:/|per)\s*s(?:ec(?:ond)?)?|(?:/|per)\s*s(?:ec(?:ond)?)?\b)", src, flags=re.I)
    ch = re.findall(r"(?<![\d.])" + NUM + r"(?:\s*-\s*" + NUM + r")?\s*(?:ch/s|chirps?\s*(?:/|per)\s*s(?:ec(?:ond)?)?)", src, flags=re.I)
    flat = [a if not b else str((float(a) + float(b)) / 2) for a, b in rates]
    d["rates_per_s"] = ";".join(flat)
    d["chirp_rates"] = ";".join(a if not b else str((float(a) + float(b)) / 2) for a, b in ch)
    d["rate_first"] = float(flat[0]) if flat else None
    d["rate_is_at_25"] = bool(at25) and bool(flat)
    temps = re.findall(NUM + r"\s*(?:°|deg|Â°)\s*C", song, flags=re.I)
    d["temps_C"] = ";".join(temps)
    khz = re.findall(NUM + r"\s*kHz", song, flags=re.I)
    d["kHz"] = ";".join(khz)
    d["kHz_first"] = float(khz[0]) if khz else None
    d["pattern"] = ";".join(sorted(set(w.lower() for w in re.findall(r"\b(trill|chirp|buzz|tick|lisp|rattle|click|zip|continuous|intermittent)\w*", song, flags=re.I))))
    return d


def main() -> None:
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for group, lst in LISTS.items():
        links = species_links(get(BASE + lst, out_dir))
        print(f"{group}: {len(links)} species pages", flush=True)
        for subfam, page, name in links:
            try:
                txt = strip(get(BASE + page, out_dir))
            except Exception as e:
                print(f"  {page} failed: {e}", flush=True)
                continue
            d = parse_song(txt)
            d.update(group=group, subfamily=subfam, page=page, name=name)
            rows.append(d)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out_dir, "sina_songs.csv"), index=False)
    print(f"{len(S)} species; with a rate: {S.rate_first.notna().sum()}, with kHz: {S.kHz_first.notna().sum()}, "
          f"with both: {(S.rate_first.notna() & S.kHz_first.notna()).sum()}")


if __name__ == "__main__":
    main()
