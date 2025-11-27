# generate_pairs.py
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import csv

random.seed(1234)
np.random.seed(1234)

# --------- CONFIG ----------
OUT_DIR = "dataset_pairs"
# 100 chữ Hán: từ đơn giản đến phức tạp (tăng từ 50 lên 100)
CHARS = [
    # Đơn giản (1-4 nét) - 30 chữ
    "一","二","三","十","人","入","八","九","七","力",
    "刀","了","大","天","木","火","水","土","山","口",
    "日","月","田","目","石","竹","耳","羊","米","舌",
    # Trung bình (5-8 nét) - 40 chữ
    "中","国","学","好","手","心","你","我","他","们",
    "来","去","不","是","有","可","在","会","到","时",
    "白","百","生","用","年","行","同","先","回","名",
    "花","草","茶","语","话","读","请","问","谢","对",
    # Phức tạp (9+ nét) - 30 chữ
    "想","爱","意","感","龍","變","鬱","體","聽","說",
    "讀","寫","課","題","學","業","樂","歡","漢","字",
    "難","題","練","習","複","雜","經","驗","選","擇"
]
IMG_SIZE = 128
POS_PER_CHAR = 150  # Giảm từ 200 xuống 150 (tổng 100*150*2 = 30k cặp)
NEG_PER_CHAR = 150
# ---------------------------

# Auto-detect Chinese font on Windows/Linux
def find_chinese_font():
    possible_fonts = [
        r"C:\Windows\Fonts\simkai.ttf",      # KaiTi
        r"C:\Windows\Fonts\kaiu.ttf",         # KaiTi Unicode
        r"C:\Windows\Fonts\msyh.ttc",         # Microsoft YaHei (fallback)
        r"C:\Windows\Fonts\simsun.ttc",       # SimSun (fallback)
        "/usr/share/fonts/truetype/arphic/ukai.ttc",  # Linux
        "/System/Library/Fonts/STHeiti Medium.ttc",   # macOS
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            print(f"Found font: {font_path}")
            return font_path
    raise FileNotFoundError("No Chinese font found. Please install KaiTi or SimSun font.")

FONT_PATH = find_chinese_font()

os.makedirs(OUT_DIR, exist_ok=True)
img_dir = os.path.join(OUT_DIR, "images")
os.makedirs(img_dir, exist_ok=True)
csv_path = os.path.join(OUT_DIR, "pairs.csv")

font = ImageFont.truetype(FONT_PATH, size=100)

def render_char(ch, size=IMG_SIZE, font=font, bgcolor=255, fgcolor=0):
    img = Image.new("L", (size, size), color=bgcolor)
    draw = ImageDraw.Draw(img)
    # Use textbbox instead of deprecated textsize
    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # center text
    draw.text(((size-w)/2, (size-h)/2 - bbox[1]), ch, font=font, fill=fgcolor)
    
    # Make strokes thicker to simulate handwriting
    arr = np.array(img)
    # Apply dilation to thicken strokes (simulates pen width 3-5px)
    kernel = np.ones((3, 3), np.uint8)
    arr = cv2.dilate(255 - arr, kernel, iterations=1)
    arr = 255 - arr
    img = Image.fromarray(arr)
    
    return img

def augment_positive(img):
    # Simulate natural handwriting variations
    # 1. Random stroke thickness (simulate pen pressure)
    if random.random() < 0.3:
        arr = np.array(img)
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.5:
            # Thicken strokes
            arr = cv2.dilate(255 - arr, kernel, iterations=1)
            arr = 255 - arr
        img = Image.fromarray(arr)
    
    # 2. Small rotation (writing angle variation)
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    
    # 3. Translate (position variation)
    tx = random.uniform(-8, 8)
    ty = random.uniform(-8, 8)
    M = (1, 0, tx, 0, 1, ty)
    img = img.transform(img.size, Image.AFFINE, M, resample=Image.BICUBIC, fillcolor=255)
    
    # 4. Slight scale variation
    if random.random() < 0.3:
        scale = random.uniform(0.95, 1.05)
        new_size = int(img.size[0] * scale)
        img = img.resize((new_size, new_size), Image.BICUBIC)
        # Crop or pad back to original size
        if new_size > img.size[0]:
            img = img.crop((
                (new_size - img.size[0]) // 2,
                (new_size - img.size[0]) // 2,
                (new_size + img.size[0]) // 2,
                (new_size + img.size[0]) // 2
            ))
        else:
            new_img = Image.new('L', (img.size[0], img.size[0]), 255)
            offset = (img.size[0] - new_size) // 2
            new_img.paste(img, (offset, offset))
            img = new_img
    
    # 5. Light blur (simulate hand tremor)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    # 6. Light noise
    if random.random() < 0.15:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 3, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    return img

def augment_negative(img):
    # heavy distortions: rotate large, shear, stretch, heavy noise
    angle = random.uniform(-45, 45)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    # shear / stretch using opencv
    arr = np.array(img)
    h, w = arr.shape
    pts1 = np.float32([[0,0],[w,0],[0,h]])
    dx = random.uniform(-0.3,0.3)*w
    dy = random.uniform(-0.3,0.3)*h
    pts2 = np.float32([[dx,dy],[w+dx*0.3,0+dy*0.2],[0+dx*0.6,h+dy*0.6]])
    M = cv2.getAffineTransform(pts1, pts2)
    arr = cv2.warpAffine(arr, M, (w,h), borderValue=255)
    # heavy noise
    noise = np.random.normal(0, 20, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img

# Pre-generate base images (rendered font)
base_imgs = {}
for ch in CHARS:
    base_imgs[ch] = render_char(ch)

# Now create pairs and write CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["img_a","img_b","label"])
    idx = 0
    for ch in CHARS:
        # positive pairs
        for i in range(POS_PER_CHAR):
            img_a = base_imgs[ch]
            img_b = augment_positive(base_imgs[ch])
            fn_a = f"img_{idx:06d}_a.png"; idx += 1
            fn_b = f"img_{idx:06d}_b.png"; idx += 1
            img_a.save(os.path.join(img_dir, fn_a))
            img_b.save(os.path.join(img_dir, fn_b))
            writer.writerow([fn_a, fn_b, 1])
        # negative pairs (mix of other chars and heavy-distorted same char)
        other_chars = CHARS.copy()
        other_chars.remove(ch)
        for i in range(NEG_PER_CHAR):
            if random.random() < 0.6:
                # different char
                ch2 = random.choice(other_chars)
                img_a = base_imgs[ch]
                img_b = augment_positive(base_imgs[ch2])  # keep positive-like other char
            else:
                # same char but heavily distorted
                img_a = base_imgs[ch]
                img_b = augment_negative(base_imgs[ch])
            fn_a = f"img_{idx:06d}_a.png"; idx += 1
            fn_b = f"img_{idx:06d}_b.png"; idx += 1
            img_a.save(os.path.join(img_dir, fn_a))
            img_b.save(os.path.join(img_dir, fn_b))
            writer.writerow([fn_a, fn_b, 0])

print("Finished. CSV at:", csv_path)
