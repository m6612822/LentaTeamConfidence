"""Render docs/presentation.html -> docs/presentation.pdf.

No Chrome/weasyprint here; uses the available Firefox ESR --screenshot +
Pillow. Single source of truth: the live deck. We derive a static
all-slides variant (JS stripped, each slide a 1280x720 page), screenshot
the full page, slice into 11 frames, assemble a landscape PDF.

  python scripts/make_pdf.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "presentation.html")
PRINT_HTML = os.path.join(ROOT, "docs", "_presentation_print.html")
SHOT = os.path.join(ROOT, "docs", "_deck_full.png")
PDF = os.path.join(ROOT, "docs", "presentation.pdf")
W, H = 1280, 720

PRINT_CSS = """
<style id="printcss">
  html,body{overflow:visible!important;height:auto!important;
    width:%dpx!important;
    -webkit-print-color-adjust:exact!important;
    print-color-adjust:exact!important}
  .bar,.hint{display:none!important}
  .slide{position:relative!important;inset:auto!important;
    display:flex!important;flex-direction:column!important;
    align-items:center!important;justify-content:center!important;
    opacity:1!important;width:%dpx!important;height:%dpx!important;
    padding:64px 110px!important;          /* fixed: vh would use 7920px */
    page-break-after:always;overflow:hidden}
  footer{position:absolute!important;left:0;right:0;bottom:0}
</style>
""" % (W, W, H)


def build_print_html():
    html = open(SRC, encoding="utf-8").read()
    html = re.sub(r"<script>.*?</script>", "", html, flags=re.S)
    html = html.replace("</head>", PRINT_CSS + "</head>")
    with open(PRINT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    return len(re.findall(r'<section class="slide', html))


def screenshot(n):
    if os.path.exists(SHOT):
        os.remove(SHOT)
    prof = tempfile.mkdtemp(prefix="ffpdf_")
    try:
        cmd = ["firefox", "--headless", "--no-remote", "--profile", prof,
               "--window-size", f"{W},{n * H}",
               f"--screenshot={SHOT}", "file://" + PRINT_HTML]
        subprocess.run(cmd, check=True, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    if not os.path.exists(SHOT):
        sys.exit("firefox screenshot failed")


def assemble():
    from PIL import Image
    im = Image.open(SHOT).convert("RGB")
    iw, ih = im.size
    if iw != W:                       # normalise width if DPR differs
        im = im.resize((W, round(ih * W / iw)))
        iw, ih = im.size
    n = max(1, round(ih / H))
    pages = []
    for i in range(n):
        y0, y1 = i * H, min(ih, (i + 1) * H)
        crop = im.crop((0, y0, W, y1))
        if crop.size[1] != H:
            cv = Image.new("RGB", (W, H), (14, 17, 22))
            cv.paste(crop, (0, 0))
            crop = cv
        pages.append(crop)
    pages[0].save(PDF, save_all=True, append_images=pages[1:],
                  resolution=96.0)
    print(f"{n} slides -> {PDF}  ({os.path.getsize(PDF)//1024} KB)")
    for p in (SHOT, PRINT_HTML):
        os.path.exists(p) and os.remove(p)


if __name__ == "__main__":
    n = build_print_html()
    screenshot(n)
    assemble()
