import cv2
import numpy as np
import os
import glob
from fpdf import FPDF
from skimage.util import random_noise
import novel_core 

INPUT_DIR = "grayscale_normalized"
OUTPUT_DIR = "results_self_recovery"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Content-Based Self-Recovery Analysis', 0, 1, 'C')
        self.ln(5)

    def add_row(self, orig, wm, att, tmap, rec, title, wpsnr, rec_psnr):
        # Save high-res PNGs
        names = ["t_orig.png", "t_wm.png", "t_att.png", "t_map.png", "t_rec.png"]
        imgs = [orig, wm, att, tmap, rec]
        for n, i in zip(names, imgs): cv2.imwrite(os.path.join(OUTPUT_DIR, n), i)

        if self.get_y() > 220: self.add_page()

        self.set_font('Arial', 'B', 10)
        self.cell(0, 6, title, 0, 1)
        self.set_font('Courier', '', 9)
        # Note: We now show Recovered Image Quality (Rec_PSNR), not just Watermark Bits
        self.cell(0, 5, f"WPSNR (Imperceptibility): {wpsnr:.2f} dB  |  Rec_PSNR (Recovery Quality): {rec_psnr:.2f} dB", 0, 1)

        y, x, w = self.get_y() + 2, 10, 36
        labels = ["Original", "Watermarked", "Attacked", "Tamper Map", "Recovered"]
        
        for i, name in enumerate(names):
            self.image(os.path.join(OUTPUT_DIR, name), x=x+(i*(w+2)), y=y, w=w)
            self.set_xy(x+(i*(w+2)), y+w+1)
            self.set_font('Arial', '', 7)
            self.cell(w, 4, labels[i], 0, 0, 'C')
        self.ln(w + 10)

class Tester:
    def psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))

    # Distinct Attacks
    def attack_text(self, img):
        res = img.copy()
        cv2.putText(res, 'HACKED', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 8)
        cv2.putText(res, 'HACKED', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        return res

    def attack_crop(self, img):
        res = img.copy()
        h, w = img.shape[:2]
        res[h//2:, :] = 0 # Bottom half black
        return res

    def attack_splicing(self, img):
        res = img.copy()
        patch = img[0:100, 0:100]
        res[100:200, 100:200] = patch # Paste top-left to center
        return res

    def attack_noise(self, img):
        n = random_noise(img, mode='s&p', amount=0.05)
        return (255*n).astype(np.uint8)

    def run(self):
        pdf = ReportPDF()
        pdf.set_auto_page_break(True, margin=15)
        images = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))

        attacks = [
            ("Text Insertion", self.attack_text),
            ("Half-Crop", self.attack_crop),
            ("Splicing (Copy-Move)", self.attack_splicing),
            ("Salt & Pepper Noise", self.attack_noise),
            ("JPEG (Q=50)", lambda x: cv2.imdecode(cv2.imencode('.jpg', x, [int(cv2.IMWRITE_JPEG_QUALITY), 50])[1], 1))
        ]

        for path in images:
            name = os.path.basename(path)
            print(f"Processing {name}...")
            
            wm_path = os.path.join(OUTPUT_DIR, "wm.png")
            novel_core.embed(path, wm_path)
            
            orig = cv2.imread(path)
            wm = cv2.imread(wm_path)
            if wm is None: continue
            
            wpsnr = self.psnr(orig, wm)
            pdf.add_page()
            pdf.cell(0,10,f"Image: {name}",0,1)

            for title, func in attacks:
                att = func(wm)
                att_path = os.path.join(OUTPUT_DIR, "att.png")
                cv2.imwrite(att_path, att)

                rec_path, map_path = os.path.join(OUTPUT_DIR, "rec.png"), os.path.join(OUTPUT_DIR, "map.png")
                novel_core.recover_and_verify(att_path, rec_path, map_path)
                
                rec = cv2.imread(rec_path)
                tmap = cv2.imread(map_path)
                
                # Calculate Recovery PSNR (Quality of the restored image vs Original)
                rec_psnr = self.psnr(orig, rec)
                
                print(f"  {title:20} | WPSNR: {wpsnr:.1f} | Rec_PSNR: {rec_psnr:.1f}")
                pdf.add_row(orig, wm, att, tmap, rec, title, wpsnr, rec_psnr)

        pdf.output("Self_Recovery_Report.pdf")
        print("Done.")

if __name__ == "__main__":
    t = Tester()
    t.run()