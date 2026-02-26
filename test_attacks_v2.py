"""
Comprehensive Image Forgery Attack Suite - Fast OpenCV Grid PDF Report.
Tests all standard image forgery attacks with grid-format visual output.
"""

import cv2
import numpy as np
import os, glob
from fpdf import FPDF
from skimage.metrics import structural_similarity as ssim
import robust_watermark

INPUT_DIR  = "grayscale_normalized"
OUTPUT_DIR = "results_v6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACKS = [
    ("JPEG Q=90",       ("jpeg", 90)),
    ("JPEG Q=70",       ("jpeg", 70)),
    ("JPEG Q=50",       ("jpeg", 50)),
    ("JPEG Q=30",       ("jpeg", 30)),
    ("Gauss Noise s=5", ("gnoise", 5)),
    ("Gauss Noise s=10",("gnoise", 10)),
    ("Gauss Noise s=20",("gnoise", 20)),
    ("S&P Noise 1%",    ("sp", 0.01)),
    ("S&P Noise 5%",    ("sp", 0.05)),
    ("Gauss Blur 3x3",  ("gblur", 3)),
    ("Gauss Blur 5x5",  ("gblur", 5)),
    ("Median 3x3",      ("median", 3)),
    ("Median 5x5",      ("median", 5)),
    ("Mean Filter 3x3", ("mean", 3)),
    ("Sharpening",      ("sharpen", None)),
    ("Motion Blur",     ("motion", 15)),
    ("Hist Equalize",   ("histeq", None)),
    ("Brightness +30",  ("bright", 30)),
    ("Brightness -30",  ("bright", -30)),
    ("Contrast x1.5",   ("contrast", 1.5)),
    ("Gamma 0.7",       ("gamma", 0.7)),
    ("Gamma 1.5",       ("gamma", 1.5)),
    ("Scale 50%",       ("scale", 0.5)),
    ("Scale 75%",       ("scale", 0.75)),
    ("Scale 200%",      ("scale", 2.0)),
    ("Rotate 5 deg",    ("rotate", 5)),
    ("Rotate 10 deg",   ("rotate", 10)),
    ("Rotate 45 deg",   ("rotate", 45)),
    ("Rotate 90 deg",   ("rotate", 90)),
    ("Crop 10%",        ("crop", 0.10)),
    ("Crop 20%",        ("crop", 0.20)),
    ("Crop 30%",        ("crop", 0.30)),
    ("Copy-Move",       ("copymove", None)),
]


def apply_attack(img, spec):
    if spec is None: return img.copy()
    kind, param = spec
    if kind == "jpeg":
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), param])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if kind == "gnoise":
        return np.clip(img.astype(np.float64) + np.random.normal(0, param, img.shape), 0, 255).astype(np.uint8)
    if kind == "sp":
        out = img.copy(); n = int(param * img.size)
        coords = tuple(np.random.randint(0, d, n) for d in img.shape); out[coords] = 255
        coords = tuple(np.random.randint(0, d, n) for d in img.shape); out[coords] = 0
        return out
    if kind == "gblur":  return cv2.GaussianBlur(img, (param, param), 0)
    if kind == "median": return cv2.medianBlur(img, param)
    if kind == "mean":   return cv2.blur(img, (param, param))
    if kind == "sharpen":
        return cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32))
    if kind == "motion":
        k = np.zeros((param, param), dtype=np.float32); k[param//2, :] = 1.0/param
        return cv2.filter2D(img, -1, k)
    if kind == "histeq": return cv2.equalizeHist(img)
    if kind == "bright": return np.clip(img.astype(np.int16) + param, 0, 255).astype(np.uint8)
    if kind == "contrast": return np.clip(img.astype(np.float64) * param, 0, 255).astype(np.uint8)
    if kind == "gamma":
        lut = np.array([((i/255.0)**param)*255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)
    if kind == "scale":
        h, w = img.shape[:2]
        s = cv2.resize(img, (int(w*param), int(h*param)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "rotate":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), param, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=0)
    if kind == "crop":
        out = img.copy(); h, w = out.shape[:2]
        ch, cw = int(h*param), int(w*param)
        sy, sx = (h-ch)//2, (w-cw)//2
        out[sy:sy+ch, sx:sx+cw] = 0; return out
    if kind == "copymove":
        out = img.copy(); h, w = out.shape[:2]
        bh, bw = h//4, w//4
        out[h//2-bh//2:h//2+bh//2, w//2-bw//2:w//2+bw//2] = out[0:bh, 0:bw]
        return out
    return img.copy()


def calc_metrics(orig, rec):
    if orig.shape != rec.shape:
        rec = cv2.resize(rec, (orig.shape[1], orig.shape[0]))
    mse = np.mean((orig.astype(float) - rec.astype(float)) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr, ssim(orig, rec)


def resize_thumb(img, size=96):
    """Resize to thumbnail for grid."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def put_text(canvas, text, x, y, scale=0.35, color=0, thickness=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def make_grid(img_name, original, watermarked, psnr_w, ssim_w, results, out_path):
    """Fast OpenCV grid: 4 columns (Attacked, TamperMap, Recovered, Metrics) per attack row."""
    T = 96   # thumbnail size
    PAD = 4
    LABEL_W = 120  # width for attack name label
    METRIC_W = 100
    ROW_H = T + PAD

    n = len(results)
    # Header height
    HEADER_H = T + 30

    cols = LABEL_W + 3 * (T + PAD) + METRIC_W + PAD * 2
    rows = HEADER_H + n * ROW_H + PAD

    canvas = np.ones((rows, cols), dtype=np.uint8) * 240

    # Header: Original + Watermarked
    put_text(canvas, f"{img_name}  |  Watermarked PSNR={psnr_w:.1f} SSIM={ssim_w:.4f}", 
             PAD, 14, scale=0.4, thickness=1)

    x = LABEL_W
    ot = resize_thumb(original, T)
    canvas[20:20+T, x:x+T] = ot
    put_text(canvas, "Original", x+10, 18, scale=0.3)

    x += T + PAD
    wt = resize_thumb(watermarked, T)
    canvas[20:20+T, x:x+T] = wt
    put_text(canvas, "Watermarked", x+5, 18, scale=0.3)

    # Column headers
    y_start = HEADER_H - 2
    put_text(canvas, "Attack", PAD, y_start, scale=0.3, thickness=1)
    put_text(canvas, "Attacked", LABEL_W + T//4, y_start, scale=0.3, thickness=1)
    put_text(canvas, "Tamper Map", LABEL_W + T + PAD + T//6, y_start, scale=0.3, thickness=1)
    put_text(canvas, "Recovered", LABEL_W + 2*(T+PAD) + T//6, y_start, scale=0.3, thickness=1)
    put_text(canvas, "PSNR/RPSNR/SSIM", LABEL_W + 3*(T+PAD), y_start, scale=0.3, thickness=1)

    # Draw separator
    cv2.line(canvas, (0, HEADER_H), (cols, HEADER_H), 150, 1)

    for i, r in enumerate(results):
        y = HEADER_H + i * ROW_H + PAD//2

        # Attack name
        put_text(canvas, r['name'], PAD, y + T//2 + 4, scale=0.3)

        # Attacked thumbnail
        x = LABEL_W
        at_thumb = resize_thumb(r['attacked'], T)
        canvas[y:y+T, x:x+T] = at_thumb

        # Tamper map thumbnail
        x += T + PAD
        tm_thumb = resize_thumb(r['tmap'], T)
        canvas[y:y+T, x:x+T] = tm_thumb

        # Recovered thumbnail
        x += T + PAD
        rc_thumb = resize_thumb(r['recovered'], T)
        canvas[y:y+T, x:x+T] = rc_thumb

        # Metrics text
        x += T + PAD
        put_text(canvas, f"PSNR: {r['psnr_att']:.1f}", x, y + T//3, scale=0.3)
        put_text(canvas, f"RPSNR: {r['psnr_rec']:.1f}", x, y + T//3 + 14, scale=0.3)
        put_text(canvas, f"SSIM: {r['ssim_rec']:.4f}", x, y + T//3 + 28, scale=0.3)

        # Row separator
        cv2.line(canvas, (0, y + T + PAD//2), (cols, y + T + PAD//2), 220, 1)

    cv2.imwrite(out_path, canvas)


def run_test_suite():
    images = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.tiff")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )
    if not images: print(f"No images in {INPUT_DIR}/"); return

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=10)

    # Title page
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.ln(20)
    pdf.cell(0, 10, "Robust Self-Recovery Watermarking", ln=True, align='C')
    pdf.cell(0, 10, "Comprehensive Forgery Attack Report", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.ln(6)
    pdf.cell(0, 6, f"Images: {len(images)}  |  Attacks: {len(ATTACKS)}", ln=True, align='C')
    pdf.cell(0, 6, "Method: 1-LSB Spatial Embedding + DCT Recovery", ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "All Attacks:", ln=True)
    pdf.set_font("Arial", "", 8)
    col = 0
    for i, (a, _) in enumerate(ATTACKS):
        if col == 0: pdf.set_x(15)
        pdf.cell(45, 5, f"{i+1}. {a}")
        col += 1
        if col >= 4: pdf.ln(); col = 0
    if col > 0: pdf.ln()

    all_results = []

    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing {img_name}...")

        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None: continue
        h = (original.shape[0]//8)*8; w = (original.shape[1]//8)*8
        original = original[:h, :w]

        wat_path = os.path.join(OUTPUT_DIR, f"watermarked_{img_name}.png")
        robust_watermark.embed(img_path, wat_path)
        watermarked = cv2.imread(wat_path, cv2.IMREAD_GRAYSCALE)[:h, :w]

        psnr_w, ssim_w = calc_metrics(original, watermarked)
        print(f"  Watermarked: PSNR={psnr_w:.2f}, SSIM={ssim_w:.4f}")

        attack_data = []
        for aname, aspec in ATTACKS:
            safe = aname.replace(" ","_").replace("=","").replace("&","n").replace(".","").replace("%","pct")
            attacked = apply_attack(watermarked, aspec)
            if attacked.shape != original.shape:
                attacked = cv2.resize(attacked, (original.shape[1], original.shape[0]))

            att_path = os.path.join(OUTPUT_DIR, f"att_{safe}_{img_name}.png")
            cv2.imwrite(att_path, attacked)
            psnr_a, ssim_a = calc_metrics(original, attacked)

            rec_path = os.path.join(OUTPUT_DIR, f"rec_{safe}_{img_name}.png")
            result = robust_watermark.recover(att_path, rec_path, return_tamper_map=True)
            tmap = result[1] if isinstance(result, tuple) else np.zeros_like(original)

            recovered = cv2.imread(rec_path, cv2.IMREAD_GRAYSCALE)
            if recovered.shape != original.shape:
                recovered = cv2.resize(recovered, (original.shape[1], original.shape[0]))
            psnr_r, ssim_r = calc_metrics(original, recovered)

            print(f"  {aname:18s}: PSNR={psnr_a:.2f}, RPSNR={psnr_r:.2f}, SSIM={ssim_r:.4f}")

            attack_data.append({
                'name': aname, 'attacked': attacked, 'tmap': tmap,
                'recovered': recovered, 'psnr_att': psnr_a, 'ssim_att': ssim_a,
                'psnr_rec': psnr_r, 'ssim_rec': ssim_r,
            })

        all_results.append((img_name, psnr_w, ssim_w, attack_data))

        # Generate fast OpenCV grid
        grid_path = os.path.join(OUTPUT_DIR, f"grid_{img_name}.png")
        make_grid(img_name, original, watermarked, psnr_w, ssim_w, attack_data, grid_path)
        print(f"  Grid saved: {grid_path}")

        # PDF: grid image page
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, f"{img_name} - All Attacks Visual Grid", ln=True, align='C')
        pdf.set_font("Arial", "", 8)
        pdf.cell(0, 5, f"Watermarked PSNR: {psnr_w:.2f} dB  |  SSIM: {ssim_w:.4f}", ln=True, align='C')
        pdf.ln(1)
        try:
            pdf.image(grid_path, x=5, y=pdf.get_y(), w=200)
        except Exception as e:
            pdf.cell(0, 6, f"(error: {e})", ln=True)

        # PDF: metrics table page
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, f"{img_name} - Metrics Table", ln=True, align='C')
        pdf.ln(2)
        cw = [48, 25, 22, 25, 22]
        pdf.set_font("Arial", "B", 8)
        for ci, ht in enumerate(["Attack", "PSNR", "SSIM", "RPSNR", "RSSIM"]):
            pdf.cell(cw[ci], 6, ht, 1, align='C')
        pdf.ln()
        for r in attack_data:
            pdf.set_font("Arial", "", 7)
            pdf.cell(cw[0], 5, r['name'], 1)
            pdf.cell(cw[1], 5, f"{r['psnr_att']:.2f}", 1, align='C')
            pdf.cell(cw[2], 5, f"{r['ssim_att']:.4f}", 1, align='C')
            pdf.cell(cw[3], 5, f"{r['psnr_rec']:.2f}", 1, align='C')
            pdf.cell(cw[4], 5, f"{r['ssim_rec']:.4f}", 1, align='C')
            pdf.ln()

    # Summary page
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Summary: Average Metrics Across All Images", ln=True, align='C')
    pdf.ln(4)
    cw = [48, 25, 22, 25, 22]
    pdf.set_font("Arial", "B", 8)
    for ci, ht in enumerate(["Attack", "Avg PSNR", "Avg SSIM", "Avg RPSNR", "Avg RSSIM"]):
        pdf.cell(cw[ci], 6, ht, 1, align='C')
    pdf.ln()
    for ai, (an, _) in enumerate(ATTACKS):
        ps = [r[3][ai]['psnr_att'] for r in all_results]
        ss = [r[3][ai]['ssim_att'] for r in all_results]
        rp = [r[3][ai]['psnr_rec'] for r in all_results]
        rs = [r[3][ai]['ssim_rec'] for r in all_results]
        pdf.set_font("Arial", "", 7)
        pdf.cell(cw[0], 5, an, 1)
        pdf.cell(cw[1], 5, f"{np.mean(ps):.2f}", 1, align='C')
        pdf.cell(cw[2], 5, f"{np.mean(ss):.4f}", 1, align='C')
        pdf.cell(cw[3], 5, f"{np.mean(rp):.2f}", 1, align='C')
        pdf.cell(cw[4], 5, f"{np.mean(rs):.4f}", 1, align='C')
        pdf.ln()

    # Per-image watermark quality
    pdf.ln(6)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Watermark Quality (Original vs Watermarked)", ln=True, align='C')
    pdf.ln(2)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(55, 6, "Image", 1, align='C')
    pdf.cell(28, 6, "PSNR (dB)", 1, align='C')
    pdf.cell(22, 6, "SSIM", 1, align='C')
    pdf.ln()
    for name, pw, sw, _ in all_results:
        pdf.set_font("Arial", "", 8)
        pdf.cell(55, 5, name, 1)
        pdf.cell(28, 5, f"{pw:.2f}", 1, align='C')
        pdf.cell(22, 5, f"{sw:.4f}", 1, align='C')
        pdf.ln()

    report = "Final_Attack_Report_v2.pdf"
    pdf.output(report)
    print(f"\n--- Report: {report} ---")
    print(f"--- Grids saved in {OUTPUT_DIR}/ ---")


if __name__ == "__main__":
    run_test_suite()
