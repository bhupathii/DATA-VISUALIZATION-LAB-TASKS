import os
import re

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agriculture_crop_yield.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(CSV_PATH)
    cols = [
        "State",
        "Crop_Type",
        "Season",
        "Climate_Zone",
        "Soil_Type",
        "Irrigation_Type",
        "Pest_Infestation_Level",
        "Disease_Incidence",
    ]
    parts = []
    for c in cols:
        if c in df.columns:
            parts.extend(df[c].astype(str).tolist())

    # Optional: include your details so TagCrowd shows them
    # Edit these or set environment variables before running
    name = os.getenv("USER_NAME", "")
    college = os.getenv("USER_COLLEGE", "")
    if name:
        parts.append(name)
    if college:
        parts.append(college)

    text = clean_text("\n".join(parts))
    os.makedirs(OUT_DIR, exist_ok=True)

    # 7a) TagCrowd input file
    tagcrowd_txt = os.path.join(OUT_DIR, "tagcrowd_input.txt")
    with open(tagcrowd_txt, "w", encoding="utf-8") as f:
        f.write(text)

    # Local WordCloud image
    wc = WordCloud(width=1200, height=800, background_color="white", max_words=300).generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    wordcloud_img = os.path.join(OUT_DIR, "wordcloud.png")
    plt.savefig(wordcloud_img, dpi=200)
    plt.close()

    print("7a outputs:")
    print("-", tagcrowd_txt)
    print("-", wordcloud_img)


if __name__ == "__main__":
    main()


