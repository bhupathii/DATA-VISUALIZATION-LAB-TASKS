import os
import re
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agriculture_crop_yield.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentences(text: str) -> List[List[str]]:
    sentences = re.split(r"[\.!?\n]+", text)
    tokenized = []
    for s in sentences:
        tokens = [t for t in re.findall(r"[a-z0-9\-]+", s) if t]
        if tokens:
            tokenized.append(tokens)
    return tokenized


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
    text = clean_text("\n".join(parts))

    tokenized = tokenize_sentences(text)
    root = os.getenv("WORDTREE_ROOT", "corn")

    # Build sequences from root forward
    sequences: List[Tuple[str, ...]] = []
    for tokens in tokenized:
        for i, tok in enumerate(tokens):
            if tok == root:
                window = tokens[i : i + 6]
                if window:
                    sequences.append(tuple(window))

    # Build weighted directed graph
    edge_weights = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            edge_weights[(a, b)] += 1

    g = nx.DiGraph()
    for (a, b), w in edge_weights.items():
        g.add_edge(a, b, weight=w)
    if root not in g.nodes:
        g.add_node(root)

    # Simple layered layout by BFS depths
    depths = {root: 0}
    q = [root]
    while q:
        cur = q.pop(0)
        d = depths[cur]
        if d >= 4:
            continue
        for nxt in g.successors(cur):
            if nxt not in depths:
                depths[nxt] = d + 1
                q.append(nxt)

    levels = {}
    for node, d in depths.items():
        levels.setdefault(d, []).append(node)

    pos = {}
    for depth, nodes in levels.items():
        nodes = sorted(nodes)
        n = len(nodes) or 1
        for i, node in enumerate(nodes):
            pos[node] = ((i + 1) / (n + 1), -depth)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 8))
    edges = g.edges(data=True)
    if edges:
        weights = [max(0.5, d.get("weight", 1) * 0.6) for (_, _, d) in edges]
        nx.draw_networkx_edges(g, pos, alpha=0.5, width=weights, arrows=True, arrowstyle="-|>")
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color="#E8F0FE", edgecolors="#3367D6")
    nx.draw_networkx_labels(g, pos, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    out_img = os.path.join(OUT_DIR, f"wordtree_{root}.png")
    plt.savefig(out_img, dpi=200)
    plt.close()

    print("7b output:")
    print("-", out_img)


if __name__ == "__main__":
    main()


