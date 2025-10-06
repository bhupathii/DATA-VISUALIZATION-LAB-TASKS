import os
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from wordcloud import WordCloud


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def build_corpus(
    df: pd.DataFrame,
    text_columns: List[str],
    personal_details: Dict[str, str] | None = None,
) -> str:
    parts: List[str] = []
    for col in text_columns:
        if col in df.columns:
            values = df[col].astype(str).tolist()
            parts.extend(values)

    # Add some high-signal numerical/context fields as tokens as well
    for numeric_col in [
        "Year",
        "Season",
        "Climate_Zone",
        "Soil_Type",
        "Irrigation_Type",
    ]:
        if numeric_col in df.columns:
            parts.extend(df[numeric_col].astype(str).tolist())

    # Inject personal details so the cloud contains them (Task 7a)
    if personal_details:
        for key, value in personal_details.items():
            if value:
                parts.append(str(value))
                parts.append(key)

    corpus_text = "\n".join(parts)
    return corpus_text


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def export_tagcrowd_input(text: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def generate_wordcloud(text: str, out_path: str, width: int = 1200, height: int = 800) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        collocations=True,
        max_words=300,
    ).generate(text)
    plt.figure(figsize=(width / 100, height / 100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def tokenize_sentences(text: str) -> List[List[str]]:
    # Simple sentence split on punctuation and newlines
    sentences = re.split(r"[\.!?\n]+", text)
    tokenized = []
    for s in sentences:
        tokens = [t for t in re.findall(r"[a-z0-9\-]+", s.lower()) if t]
        if tokens:
            tokenized.append(tokens)
    return tokenized


def build_wordtree_sequences(tokenized_sentences: List[List[str]], root: str, max_len: int = 6) -> List[Tuple[str, ...]]:
    sequences: List[Tuple[str, ...]] = []
    for tokens in tokenized_sentences:
        for i, tok in enumerate(tokens):
            if tok == root:
                window = tokens[i : i + max_len]
                if window:
                    sequences.append(tuple(window))
    return sequences


def sequences_to_graph(sequences: List[Tuple[str, ...]]) -> nx.DiGraph:
    g = nx.DiGraph()
    edge_weights: Dict[Tuple[str, str], int] = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            edge_weights[(a, b)] += 1
    for (a, b), w in edge_weights.items():
        g.add_edge(a, b, weight=w)
    return g


def draw_wordtree(
    g: nx.DiGraph,
    root: str,
    out_path: str,
    max_depth: int = 4,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute a layered tree layout from root
    levels: Dict[int, List[str]] = defaultdict(list)

    def bfs_layers(start: str, depth_limit: int) -> Dict[str, int]:
        depths: Dict[str, int] = {start: 0}
        queue: List[str] = [start]
        while queue:
            cur = queue.pop(0)
            d = depths[cur]
            if d >= depth_limit:
                continue
            for nxt in g.successors(cur):
                if nxt not in depths:
                    depths[nxt] = d + 1
                    queue.append(nxt)
        return depths

    depths = bfs_layers(root, max_depth)
    for node, d in depths.items():
        levels[d].append(node)

    # Assign positions: x spaced across level, y by level
    pos: Dict[str, Tuple[float, float]] = {}
    for depth, nodes in levels.items():
        nodes_sorted = sorted(nodes)
        n = len(nodes_sorted) if nodes_sorted else 1
        for idx, node in enumerate(nodes_sorted):
            x = (idx + 1) / (n + 1)
            y = -depth
            pos[node] = (x, y)

    plt.figure(figsize=(12, 8))

    # Draw edges with thickness = weight
    edges = g.edges(data=True)
    if edges:
        weights = [max(0.5, d.get("weight", 1) * 0.6) for (_, _, d) in edges]
        nx.draw_networkx_edges(g, pos, alpha=0.5, width=weights, arrows=True, arrowstyle="-|>")

    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color="#E8F0FE", edgecolors="#3367D6")

    # Draw labels
    nx.draw_networkx_labels(g, pos, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "agriculture_crop_yield.csv")
    outputs_dir = os.path.join(base_dir, "outputs")

    df = load_dataset(csv_path)

    # Columns with categorical/textual content
    text_columns = [
        "State",
        "Crop_Type",
        "Season",
        "Climate_Zone",
        "Soil_Type",
        "Irrigation_Type",
        "Pest_Infestation_Level",
        "Disease_Incidence",
    ]

    # Put your details here to include them in TagCrowd/WordCloud
    personal_details = {
        # e.g., "name": "Karthik", "university": "XYZ University", "id": "12345"
    }

    corpus = build_corpus(df, text_columns, personal_details)
    corpus_clean = clean_text(corpus)

    # 7a) Export for TagCrowd
    tagcrowd_txt = os.path.join(outputs_dir, "tagcrowd_input.txt")
    export_tagcrowd_input(corpus_clean, tagcrowd_txt)

    # Also generate a WordCloud image locally
    wordcloud_img = os.path.join(outputs_dir, "wordcloud.png")
    generate_wordcloud(corpus_clean, wordcloud_img)

    # 7b) WordTree around root token (adjust as needed: "corn", "wheat", etc.)
    root_token = "corn"
    tokenized = tokenize_sentences(corpus_clean)
    sequences = build_wordtree_sequences(tokenized, root=root_token, max_len=6)
    g = sequences_to_graph(sequences)
    wordtree_img = os.path.join(outputs_dir, f"wordtree_{root_token}.png")
    if root_token not in g.nodes:
        g.add_node(root_token)
    draw_wordtree(g, root=root_token, out_path=wordtree_img, max_depth=4)

    print("Outputs written to:")
    print(f"- {tagcrowd_txt}")
    print(f"- {wordcloud_img}")
    print(f"- {wordtree_img}")


if __name__ == "__main__":
    main()


