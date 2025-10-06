## Task 7: Text Network Analysis & Visualization

This project generates insights from `agriculture_crop_yield.csv` using:
- 7a: TagCrowd (tag cloud) + local WordCloud
- 7b: WordTree visualization

### Setup
```bash
python3 -m pip install pandas matplotlib wordcloud networkx
```

### Files
- `t7a_tagcloud_wordcloud.py` — TagCrowd input + WordCloud
- `t7b_wordtree.py` — WordTree plot

---

### 7a) TagCrowd + WordCloud

Algorithm (5 simple steps):
1. Read CSV and select text columns (`State`, `Crop_Type`, `Season`, `Climate_Zone`, `Soil_Type`, `Irrigation_Type`, `Pest_Infestation_Level`, `Disease_Incidence`).
2. Concatenate values to form a corpus; optionally append your name/college from `USER_NAME`/`USER_COLLEGE`.
3. Clean text: lowercase, remove punctuation, normalize spaces.
4. Save cleaned text to `outputs/tagcrowd_input.txt` (upload to TagCrowd).
5. Generate `outputs/wordcloud.png` locally using the `wordcloud` library.

Run:
```bash
python3 "t7a_tagcloud_wordcloud.py"
```

Optional (include your details):
```bash
export USER_NAME="Karthik"; export USER_COLLEGE="Your College"
python3 "t7a_tagcloud_wordcloud.py"
```

TagCrowd:
- Open `https://tagcrowd.com` → Create
- Upload `outputs/tagcrowd_input.txt` → Adjust settings → Visualize

Outputs:
- `outputs/tagcrowd_input.txt`
- `outputs/wordcloud.png`

---

### 7b) WordTree

Algorithm (5 simple steps):
1. Read the same CSV; build a corpus from the same text columns.
2. Clean text (lowercase, remove punctuation, normalize spaces).
3. Tokenize into sentences and tokens.
4. Extract forward sequences starting from root word (default: `corn`), count edge frequencies.
5. Plot a directed graph (layers by distance from root) to `outputs/wordtree_<root>.png`.

Run (default root `corn`):
```bash
python3 "t7b_wordtree.py"
```

Change root (example `wheat`):
```bash
WORDTREE_ROOT=wheat python3 "t7b_wordtree.py"
```

Outputs:
- `outputs/wordtree_corn.png` (or root you set)

---

Notes
- You can edit columns or parameters inside each script to tailor the analysis.
- All outputs are written under the `outputs/` folder next to the scripts.


