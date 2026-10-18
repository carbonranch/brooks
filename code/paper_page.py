"""Paper pages in MBE's house style (his Figures 1-5 of 2026-10-10, runs/figures_paper/MBE_Figures/, read-only):

  * an 8.5 x 11 in page, 1 in margins, the figure centered horizontally and aligned to the top;
  * panel labels "(A) Short title" in 8 pt bold DejaVu Sans, left-aligned with the panel's axes, sentence case,
    a descriptive phrase (wrapped to the panel's width when it is a sentence);
  * body text 6 pt (ticks, axis labels, in-panel notes); 4-5 pt only where something must be tiny;
  * one legend line below the figure, about half an inch under its lowest axis: "Figure N." in 9 pt bold, then
    the figure's title in 9 pt regular (and the legend text after it, wrapped, when there is one);
  * TrueType text everywhere so he can edit it in Illustrator.

A figure script makes a PaperPage first, draws its matplotlib figure at page.size or on a wider canvas (placed at
full size, overhanging the margins, so every font size is a size on the page), labels panels with page.label(ax, "A", "Short title") and ends with page.finish(fig, out).

    page = PaperPage(number=2, title="Recorder locations and uptime")
    fig = plt.figure(figsize=page.size)
    ...
    page.label(ax, "A", "Back9")
    page.finish(fig, "analysis/callers/runs/figures_paper/F02_sites")

Outputs: <out>.pdf and <out>.png (the pages), <out>_content.pdf / .png (the bare figure).
finish() saves the content figure (PDF with TrueType text, PNG at PAGE_DPI), places it on a letter page with pypdf
(vector text stays editable), draws the legend line with matplotlib, and composes the PNG page with PIL.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

PAGE_W, PAGE_H, MARGIN = 8.5, 11.0, 1.0           # inches
LEGEND_PT, LABEL_PT, BODY_PT, LEADING = 9.0, 8.0, 6.0, 1.3
LEGEND_GAP_IN = 0.45                                # from the figure's lowest ink to the legend line
LEGEND_CHARS = 112                                  # wrap width at 9 pt DejaVu Sans over 6.5 in
PAGE_DPI = 200
FAMILY = "DejaVu Sans"


def house_style() -> None:
    """matplotlib defaults for the body of a figure: 6 pt text, thin spines, TrueType output."""
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": FAMILY,
        "font.size": BODY_PT, "axes.labelsize": BODY_PT, "axes.titlesize": BODY_PT, "xtick.labelsize": BODY_PT, "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
    })


def wrap_legend(text: str) -> list[str]:
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, LEGEND_CHARS) or [""]
    return lines


class PaperPage:
    def __init__(self, legend: str = "", width_in: float | None = None, height_in: float | None = None, title: str | None = None,
                 number: int | str | None = None):
        """number and title make the legend line "Figure N. Title"; legend is the longer text that follows the title
        (may be empty while MBE writes it). Without width_in / height_in the figure is made at the page's own
        content size (scale 1, so every font size is a size on the page); with them it is scaled to fit."""
        house_style()
        self.number, self.title, self.legend = number, title, legend
        self.legend_lines = self._lines()
        self.legend_h = (len(self.legend_lines) * LEGEND_PT * LEADING) / 72 if self.legend_lines else 0.0
        self.avail_w = PAGE_W - 2 * MARGIN
        self.avail_h = PAGE_H - 2 * MARGIN - self.legend_h - (LEGEND_GAP_IN if self.legend_lines else 0.0)
        self.width_in = width_in if width_in is not None else self.avail_w
        self.height_in = height_in if height_in is not None else self.avail_h
        # scale 1 whenever the figure fits the page's height: a canvas wider than the text width is placed at full
        # size, centered, overhanging the margins (its extra width only holds labels), so every point size on
        # the figure is a point size on the page (MBE, 2026-10-11: "your fonts are not 8 and 6 point")
        self.scale = min(1.0, self.avail_h / self.height_in, (PAGE_W - 0.5) / self.width_in)
        self.placed_w, self.placed_h = self.width_in * self.scale, self.height_in * self.scale
        self.x0 = MARGIN + (self.avail_w - self.placed_w) / 2             # centered horizontally (may overhang the margins)
        self.y_top = PAGE_H - MARGIN                                       # aligned to the top (from the bottom edge)
        self.legend_top = None                                             # set by finish(): inches from the top edge

    # ---- text ----
    def _prefix(self) -> str:
        return f"Figure {self.number}." if self.number is not None else ""

    def _lines(self) -> list[str]:
        full = " ".join(s for s in (self._prefix(), self.title or "", self.legend or "") if s).strip()
        return wrap_legend(full) if full else []

    @property
    def size(self):
        return (self.width_in, self.height_in)

    def pt(self, points_on_page: float) -> float:
        """Font size in figure points that prints at points_on_page on the page."""
        return points_on_page / self.scale

    def label(self, ax, letter: str, text: str = "", x: float = 0.0, y: float = 1.02, color=None, chars: int | None = None):
        """'(A) Short title' in bold LABEL_PT on the page, above the axes' top-left corner (wrapped at chars)."""
        kw = {"color": color} if color is not None else {}
        body = f"({letter}) {text}".strip() if text else f"({letter})"
        if chars:
            body = textwrap.fill(body, chars)
        return ax.text(x, y, body, transform=ax.transAxes, fontsize=self.pt(LABEL_PT), fontweight="bold", va="bottom", ha="left", **kw)

    # ---- output ----
    def finish(self, fig, out: str, legend: str | None = None, content_dpi: float | None = None) -> None:
        if legend is not None and legend != self.legend:
            self.legend = legend; self.legend_lines = self._lines()
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        w, h = (float(v) for v in fig.get_size_inches())
        if abs(w - self.width_in) > 1e-6 or abs(h - self.height_in) > 1e-6:    # the figure was made at another size: place that size
            self.width_in, self.height_in = w, h
            self.scale = min(1.0, self.avail_h / h, (PAGE_W - 0.5) / w)
            self.placed_w, self.placed_h = w * self.scale, h * self.scale
            self.x0 = MARGIN + (self.avail_w - self.placed_w) / 2
        dpi = content_dpi or max(72.0, PAGE_DPI * self.scale)
        fig.savefig(out + "_content.pdf", dpi=dpi)                         # fixed size: no tight bbox, so the scale holds; rasterized artists at page resolution
        fig.savefig(out + "_content.png", dpi=dpi)
        # the legend line sits LEGEND_GAP_IN below the figure's lowest ink (never below the bottom margin)
        # the lowest axes on the figure, plus room for its tick labels and x label (the tight bbox is unreliable here:
        # hidden artists and the renderer's dpi after savefig both fooled it)
        axes = [a for a in fig.axes if a.get_visible()]
        if axes:
            low = min(a.get_position().y0 for a in axes)
            lowest = [a for a in axes if abs(a.get_position().y0 - low) < 1e-6]
            extra = 0.30 if any(a.get_xlabel() for a in lowest) else 0.18
            ink_bottom = MARGIN + ((1 - low) * fig.get_figheight() + extra) * self.scale   # inches from the top edge
        else:
            ink_bottom = MARGIN + self.placed_h
        self.legend_top = min(ink_bottom + LEGEND_GAP_IN, PAGE_H - MARGIN - self.legend_h)
        self._compose_pdf(out)
        self._compose_png(out)

    def _legend_figure(self):
        """A letter-size matplotlib figure holding the legend line(s) and nothing else: the prefix in bold."""
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        if not self.legend_lines:
            return fig
        r = fig.canvas.get_renderer()
        y = (PAGE_H - self.legend_top) / PAGE_H                            # top of the first line, figure fraction
        lh = LEGEND_PT * LEADING / 72 / PAGE_H
        for i, line in enumerate(self.legend_lines):
            x = MARGIN / PAGE_W; yy = y - i * lh
            pre = self._prefix()
            if i == 0 and pre and line.startswith(pre):
                t = fig.text(x, yy, pre, fontsize=LEGEND_PT, fontweight="bold", va="top", ha="left", family=FAMILY)
                x = t.get_window_extent(r).x1 / (PAGE_W * fig.dpi) + (LEGEND_PT * 0.28 / 72) / PAGE_W
                line = line[len(pre):].lstrip()
            fig.text(x, yy, line, fontsize=LEGEND_PT, va="top", ha="left", family=FAMILY)
        return fig

    def _compose_pdf(self, out: str) -> None:
        from pypdf import PdfReader, PdfWriter, Transformation
        import io
        buf = io.BytesIO(); lf = self._legend_figure(); lf.savefig(buf, format="pdf"); plt.close(lf); buf.seek(0)
        base = PdfReader(buf).pages[0]
        content = PdfReader(out + "_content.pdf").pages[0]
        mb = content.mediabox
        s = self.scale
        tx = self.x0 * 72 - float(mb.left) * s
        ty = (self.y_top - self.placed_h) * 72 - float(mb.bottom) * s
        base.merge_transformed_page(content, Transformation().scale(s, s).translate(tx, ty))
        w = PdfWriter(); w.add_page(base)
        with open(out + ".pdf", "wb") as fh:
            w.write(fh)

    def _compose_png(self, out: str) -> None:
        from PIL import Image, ImageDraw, ImageFont
        page = Image.new("RGB", (int(PAGE_W * PAGE_DPI), int(PAGE_H * PAGE_DPI)), "white")
        im = Image.open(out + "_content.png").convert("RGB")
        tw, th = int(round(self.placed_w * PAGE_DPI)), int(round(self.placed_h * PAGE_DPI))
        im = im.resize((tw, th), Image.LANCZOS)
        page.paste(im, (int(round(self.x0 * PAGE_DPI)), int(round((PAGE_H - self.y_top) * PAGE_DPI))))
        if self.legend_lines:
            px = int(round(LEGEND_PT / 72 * PAGE_DPI))
            font = ImageFont.truetype(font_manager.findfont(FAMILY), px)
            bold = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family=FAMILY, weight="bold")), px)
            draw = ImageDraw.Draw(page)
            line_h = LEGEND_PT * LEADING / 72 * PAGE_DPI
            y = self.legend_top * PAGE_DPI
            for i, line in enumerate(self.legend_lines):
                x = MARGIN * PAGE_DPI; pre = self._prefix()
                if i == 0 and pre and line.startswith(pre):
                    draw.text((x, y), pre, fill="black", font=bold); x += draw.textlength(pre, font=bold) + px * 0.28
                    line = line[len(pre):].lstrip()
                draw.text((x, y), line, fill="black", font=font); y += line_h
        page.save(out + ".png", dpi=(PAGE_DPI, PAGE_DPI))


if __name__ == "__main__":      # a smoke test: two panels and a legend line
    import sys
    page = PaperPage(number=0, title="A test figure", legend="(A) A sine. (B) A cosine. " * 5)
    fig, axes = plt.subplots(1, 2, figsize=(page.size[0], 2.5))
    x = np.linspace(0, 10, 200)
    for ax, fn, L in zip(axes, (np.sin, np.cos), "AB"):
        ax.plot(x, fn(x)); page.label(ax, L, "a short title")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.2)
    page.finish(fig, sys.argv[1] if len(sys.argv) > 1 else "/tmp/paper_page_test")
    print("scale", round(page.scale, 3), "placed", round(page.placed_w, 2), "x", round(page.placed_h, 2), "in; legend top", round(page.legend_top, 2))
