"""Site map and recorder uptime for the year: the whole ranch with all 27 recorders and every area labeled,
and under it a heat map of each recorder's recorded hours per day across the 13 months (Nov 2024 - Nov 2025).

(A) the whole property on MBE's drone orthomosaic (May 2024), cropped to the ranch boundary shapefile with the
    outline drawn (fallback when either is missing: USGS National Map imagery, public domain, fetched once and
    cached beside the figure), every recorder that recorded in the period as a dot at its deployment position,
    one color per area, areas labeled with their recorder count, 500 m scale bar.
(B) recorded hours per day for each recorder (rows grouped by area in (A)'s order, west to east, and colors),
    from the audio catalog (wavfiles.parquet); white = no audio that day.

Positions come from data/deployments.parquet, scoped to the (area, recorder) deployment that overlaps the period
(the recorder's own GPS is not used). A recorder moved a few meters within an area keeps its latest position.

Usage:
  python analysis/callers/site_map_figure.py --out analysis/callers/runs/figures_paper/F01b_site_map
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specdisplay as SD  # noqa: E402

PERIOD = ("2024-11-01", "2025-11-30")                     # inclusive days; 13 months
TZ = "America/Chicago"
CRS = "EPSG:32615"                                         # UTM 15N, meters
MARGIN_M = 250.0
# areas west to east (by recorder centroid), the row order of (B) and the legend order
AREAS = ["WestSideJunction", "Back9", "Copse", "House", "SecretGarden", "MidCreek", "FarNorth",
         "NorthSide", "EastCreek", "EastMeander", "Pond"]
AREA_COLORS = dict(zip(AREAS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                               "#17becf", "#bcbd22", "#7f7f7f", "#000000"]))
# label offsets (m) from each area's recorder centroid, chosen so labels do not cover dots or each other
LABEL_DXY = {"WestSideJunction": (0, 110), "Back9": (0, -130), "Copse": (0, -110), "House": (-90, 120),
             "SecretGarden": (-60, 120), "MidCreek": (120, 90), "FarNorth": (0, 110), "NorthSide": (0, 140),
             "EastCreek": (0, -130), "EastMeander": (0, 120), "Pond": (0, -110)}
USGS = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export"
# MBE's whole-ranch drone orthomosaic (May 2024, 0.42 m, UTM 15N; 81 MB, gitignored) and the ranch boundary
# shapefile (four parcels whose union is the ranch, 417 ha), both placed in orthomosaics/ on 2026-10-07
MOSAIC = "orthomosaics/CompleteSurvey_May2024_20ipp/CompleteSurvey_ortho_export_May2024_20ipp.tif"
BOUNDARY = "orthomosaics/shapefiles/shapefiles/FergusonShapes.shp"


def recorders_in_period(catalog: str) -> pd.DataFrame:
    """Recorded hours per recorder per calendar day (local) over the period, from the audio catalog."""
    W = pd.read_parquet(catalog, columns=["recorder", "recorder_group", "start_datetime", "duration"])
    W = W[W.recorder.fillna("").str.startswith("BARLT")]
    t0 = pd.Timestamp(PERIOD[0], tz=TZ); t1 = pd.Timestamp(PERIOD[1], tz=TZ) + pd.Timedelta(days=1)
    W = W[(W.start_datetime >= t0) & (W.start_datetime < t1)].copy()
    W["date"] = W.start_datetime.dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    H = W.groupby(["recorder_group", "recorder", "date"]).duration.sum().div(3600).rename("hours").reset_index()
    return H


def positions(deployments: str, pairs: pd.DataFrame) -> pd.DataFrame:
    """UTM position of each (area, recorder) pair from the deployment that overlaps the period."""
    D = pd.read_parquet(deployments)
    tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    t0 = pd.Timestamp(PERIOD[0], tz=TZ); t1 = pd.Timestamp(PERIOD[1], tz=TZ) + pd.Timedelta(days=1)
    end = D.end.fillna(pd.Timestamp("2100-01-01", tz=TZ))
    D = D[(D.begin < t1) & (end > t0)]
    rows = []
    for _, p in pairs.iterrows():
        d = D[(D.recorder_group == p.recorder_group) & (D.recorder == p.recorder)].sort_values("begin")
        if d.empty:
            print(f"  no deployment for {p.recorder} at {p.recorder_group}", file=sys.stderr); continue
        d = d.iloc[-1]
        x, y = tf.transform(d.longitude, d.latitude)
        rows.append(dict(area=p.recorder_group, recorder=p.recorder, x=x, y=y))
    return pd.DataFrame(rows)


def gpd_points(P: pd.DataFrame):
    from shapely.geometry import Point
    return [Point(x, y) for x, y in zip(P.x, P.y)]


def ranch_boundary(path: str):
    """The ranch outline from a shapefile (any CRS): the union of its polygons, in UTM."""
    import geopandas as gpd
    from shapely.ops import unary_union
    g = gpd.read_file(path).to_crs(CRS)
    return unary_union(g.geometry)


def mosaic_crop(path: str, xmin, ymin, xmax, ymax, boundary=None, max_px=3600):
    """Crop of a GeoTIFF orthomosaic in UTM covering the bbox, downsampled so the long side is at most max_px
    pixels; where a boundary polygon is given, pixels outside it are made transparent. Returns (rgba, extent) or
    None where the mosaic does not cover the bbox."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        if src.crs is None:
            raise SystemExit(f"{path} has no CRS")
        if str(src.crs) != CRS:
            raise SystemExit(f"{path} is in {src.crs}; reproject it to {CRS} first (gdalwarp -t_srs {CRS})")
        b = src.bounds
        if xmin < b.left or ymin < b.bottom or xmax > b.right or ymax > b.top:
            over = max(b.left - xmin, b.bottom - ymin, xmax - b.right, ymax - b.top)
            if over > 150:
                print(f"  mosaic {path} covers {b}, the map bbox extends {over:.0f} m beyond it; falling back to USGS imagery", file=sys.stderr)
                return None
            print(f"  map bbox trimmed by up to {over:.0f} m to the mosaic's edge", file=sys.stderr)
            xmin, ymin, xmax, ymax = max(xmin, b.left), max(ymin, b.bottom), min(xmax, b.right), min(ymax, b.top)
        win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        scale = min(1.0, max_px / max(win.width, win.height))
        h, w = int(round(win.height * scale)), int(round(win.width * scale))
        bands = [1, 2, 3, 4] if src.count >= 4 else [1, 2, 3]
        img = src.read(bands, window=win, out_shape=(len(bands), h, w), resampling=Resampling.average)
        img = np.moveaxis(img, 0, -1).astype(float)
        img = img / 255.0 if src.dtypes[0] == "uint8" else np.clip(img / max(1, np.percentile(img[..., :3], 99.5)), 0, 1)
        if img.shape[-1] == 3:
            img = np.concatenate([img, np.ones(img.shape[:2] + (1,))], axis=-1)
        if boundary is not None:
            tr = transform_from_bounds(xmin, ymin, xmax, ymax, w, h)
            outside = geometry_mask([boundary], out_shape=(h, w), transform=tr, invert=False)
            img[outside, 3] = 0.0
        return img, (xmin, xmax, ymin, ymax)


def basemap(cache: str, xmin, ymin, xmax, ymax, px_per_m=0.4):
    """USGS imagery export for the bbox (UTM), cached as PNG + JSON of its bounds."""
    meta = cache + ".json"
    if os.path.exists(cache) and os.path.exists(meta):
        b = json.load(open(meta))
        return plt.imread(cache), (b["xmin"], b["xmax"], b["ymin"], b["ymax"])
    w = int((xmax - xmin) * px_per_m); h = int((ymax - ymin) * px_per_m)
    q = dict(bbox=f"{xmin},{ymin},{xmax},{ymax}", bboxSR=CRS.split(":")[1], imageSR=CRS.split(":")[1],
             size=f"{w},{h}", format="png", f="image")
    url = USGS + "?" + urllib.parse.urlencode(q)
    print(f"fetching basemap {w}x{h} px from {USGS}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "wb").write(data)
    json.dump(dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, source=url), open(meta, "w"), indent=1)
    return plt.imread(cache), (xmin, xmax, ymin, ymax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", default="wavfiles.parquet")
    ap.add_argument("--deployments", default="data/deployments.parquet")
    ap.add_argument("--basemap", default=None, help="cached basemap PNG (default: <out dir>/basemap_usgs_ranch.png)")
    ap.add_argument("--mosaic", default=MOSAIC if os.path.exists(MOSAIC) else None,
                    help="GeoTIFF orthomosaic of the whole ranch (UTM 15N) to use instead of USGS imagery (default: %(default)s)")
    ap.add_argument("--boundary", default=BOUNDARY if os.path.exists(BOUNDARY) else None,
                    help="shapefile of the ranch boundary: the mosaic is cropped to it and the outline drawn; the map extent follows it (default: %(default)s)")
    args = ap.parse_args()
    SD.paper_fonts()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    H = recorders_in_period(args.catalog)
    pairs = H.groupby(["recorder_group", "recorder"]).hours.sum().reset_index()
    P = positions(args.deployments, pairs)
    missing = [a for a in P.area.unique() if a not in AREAS]
    assert not missing, f"areas without an order/color: {missing}"
    P["order"] = P.area.map({a: i for i, a in enumerate(AREAS)})
    P = P.sort_values(["order", "recorder"]).reset_index(drop=True)
    n_rec = len(P)
    print(f"{n_rec} recorders in {P.area.nunique()} areas; {H.hours.sum():,.0f} recorded hours over {H.date.nunique()} days")

    boundary = ranch_boundary(args.boundary) if args.boundary else None
    if boundary is not None:
        bx0, by0, bx1, by1 = boundary.bounds
        xmin, xmax = min(bx0, P.x.min()) - 60, max(bx1, P.x.max()) + 60
        ymin, ymax = min(by0, P.y.min()) - 60, max(by1, P.y.max()) + 60
        print(f"boundary: {boundary.area / 1e4:.0f} ha, {sum(boundary.contains(pt) for pt in gpd_points(P))} of {n_rec} recorders inside")
    else:
        xmin, xmax = P.x.min() - MARGIN_M, P.x.max() + MARGIN_M
        ymin, ymax = P.y.min() - MARGIN_M, P.y.max() + MARGIN_M
    img = None; credit = "imagery: USGS National Map"
    if args.mosaic:
        got = mosaic_crop(args.mosaic, xmin, ymin, xmax, ymax, boundary)
        if got is not None:
            img, ext = got; credit = "drone orthomosaic, May 2024" + (", cropped to the ranch boundary" if boundary is not None else "")
    if img is None:
        cache = args.basemap or os.path.join(os.path.dirname(args.out), "basemap_usgs_ranch.png")
        img, ext = basemap(cache, xmin, ymin, xmax, ymax)

    days = pd.date_range(PERIOD[0], PERIOD[1], freq="D")
    M = (H.pivot_table(index="recorder", columns="date", values="hours", aggfunc="sum")
         .reindex(index=P.recorder, columns=days).fillna(0.0))

    aspect = (ymax - ymin) / (xmax - xmin)
    width = 11.0
    hA = width * aspect; hB = 0.19 * n_rec + 0.9
    fig = plt.figure(figsize=(width, hA + hB + 0.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[hA, hB], hspace=0.07, left=0.11, right=0.985, top=0.965, bottom=0.06)

    # (A) map
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#f2f2ee")
    ax.imshow(img, extent=[ext[0], ext[1], ext[2], ext[3]], interpolation="bilinear", zorder=0)
    if boundary is not None:
        for poly in getattr(boundary, "geoms", [boundary]):
            bx, by = poly.exterior.xy
            ax.plot(bx, by, color="white", lw=1.0, zorder=2, path_effects=[pe.withStroke(linewidth=2.2, foreground="black", alpha=0.6)])
    halo = [pe.withStroke(linewidth=3, foreground="black", alpha=0.85)]
    for area, g in P.groupby("area", sort=False):
        c = AREA_COLORS[area]
        ax.scatter(g.x, g.y, s=34, color=c, edgecolor="white", linewidth=0.9, zorder=3)
        dx, dy = LABEL_DXY[area]
        cx, cy = g.x.mean(), g.y.mean()
        ax.annotate(f"{area} ({len(g)})", (cx, cy), (cx + dx, cy + dy), ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white", path_effects=halo, zorder=4,
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8, alpha=0.9, shrinkA=0, shrinkB=4) if abs(dy) > 60 or abs(dx) > 60 else None)
    # scale bar, 500 m, bottom left
    sx0, sy0 = xmin + 120, ymin + 110
    ax.plot([sx0, sx0 + 500], [sy0, sy0], color="white", lw=3, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=5, foreground="black")], zorder=5)
    ax.text(sx0 + 250, sy0 + 35, "500 m", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold", path_effects=halo, zorder=5)
    # north arrow, top right
    nx, ny = xmax - 140, ymax - 260
    ax.annotate("N", (nx, ny + 150), (nx, ny), ha="center", va="center", color="white", fontsize=10, fontweight="bold", path_effects=halo,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5, path_effects=[pe.withStroke(linewidth=3, foreground="black")]), zorder=5)
    ax.text(xmax - 20, ymin + 20, credit, ha="right", va="bottom", color="white", fontsize=6.5, path_effects=halo, zorder=5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_title(f"(A)  the Ferguson ranch: {n_rec} recorders in {P.area.nunique()} areas, Nov 2024 to Nov 2025",
                 loc="left", fontsize=10, fontweight="bold")

    # (B) uptime heat map
    ax = fig.add_subplot(gs[1])
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    cmap.set_under("white")
    edges = mdates.date2num(np.append(days.values, days.values[-1] + np.timedelta64(1, "D")))
    yedges = np.arange(n_rec + 1)
    pm = ax.pcolormesh(edges, yedges, M.values, cmap=cmap, vmin=0.5, vmax=24, shading="flat", rasterized=True)
    ax.set_ylim(n_rec, 0)
    ax.set_yticks(np.arange(n_rec) + 0.5)
    ax.set_yticklabels([r.replace("BARLT_0000", "") .replace("BARLT_000", "") for r in P.recorder], fontsize=7)
    for lab, area in zip(ax.get_yticklabels(), P.area):
        lab.set_color(AREA_COLORS[area])
    # area bands at left and separators
    bounds = [(g.index.min(), g.index.max() + 1) for _, g in P.groupby("order")]
    for (a, b), area in zip(bounds, AREAS):
        if b > a + 0 and a < n_rec:
            ax.axhline(b, color="white", lw=1.2)
            ax.annotate("", (edges[0], a), (edges[0], b), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", lw=5, color=AREA_COLORS[area], shrinkA=0, shrinkB=0), xycoords=("data", "data"))
            ax.text(edges[0] - 0.012 * (edges[-1] - edges[0]) - 14, (a + b) / 2, area, ha="right", va="center", fontsize=7.5,
                    color=AREA_COLORS[area], fontweight="bold", clip_on=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=7.5); ax.tick_params(axis="y", length=0)
    ax.set_xlim(edges[0], edges[-1])
    cb = fig.colorbar(pm, ax=ax, fraction=0.025, pad=0.01, extend="min")
    cb.set_label("recorded hours per day", fontsize=8); cb.ax.tick_params(labelsize=7); cb.set_ticks([0.5, 6, 12, 18, 24]); cb.set_ticklabels(["0", "6", "12", "18", "24"])
    rec_days = int((M.values > 0.5).sum())
    ax.set_title(f"(B)  recording uptime: hours of audio per recorder per day, {len(days)} days, {rec_days:,} recorder-days, "
                 f"{H.hours.sum():,.0f} h in all (white = no audio)", loc="left", fontsize=10, fontweight="bold")

    fig.savefig(args.out + ".pdf", bbox_inches="tight"); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    M.T.to_csv(args.out + "_hours_by_day.csv", float_format="%.2f")
    print(f"wrote {args.out}.pdf / .png / _hours_by_day.csv")


if __name__ == "__main__":
    main()
