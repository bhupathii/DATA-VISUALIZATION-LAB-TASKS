## Text Network Analysis (Task 7)

This project performs Text Network Analysis and Visualization on `agriculture_crop_yield.csv` using a lightweight corpus built from categorical fields.

### Minimal Python example
```python
import pandas as pd
from collections import Counter

df = pd.read_csv("agriculture_crop_yield.csv")
corpus_rows = (df["State"].astype(str) + " " + df["Crop_Type"].astype(str)).str.lower()
tokens = [t for row in corpus_rows for t in row.split()]
print(Counter(tokens).most_common(10))
```

### Simple 6-step algorithm
- **1. Select fields**: Choose text-like columns (e.g., `State`, `Crop_Type`, `Season`, `Climate_Zone`, `Soil_Type`, `Irrigation_Type`). Optionally discretize numeric fields into bins (e.g., `precip_low/med/high`).
- **2. Normalize text**: Lowercase, keep alphabetic tokens, and remove trivial stopwords.
- **3. Build corpus**: For each row, concatenate selected fields and tokenize to create one short document per row.
- **4. Tag cloud**: Aggregate token frequencies for a tag table and generate a word cloud.
- **5. Co-occurrence network**: For each row, link co-present tokens; weight edges by co-occurrence counts and detect communities (e.g., Louvain/greedy modularity).
- **6. Export and interpret**: Save `nodes/edges` CSVs, a quick network plot, a WordTree HTML, and a plain text corpus for InfraNodus. Interpret frequent terms, clusters, and bridges.

### How to run the full workflow
- Open and run `Text_Network_Analysis.ipynb` (creates outputs in `outputs/`).
- Or run headlessly (already executed here); see files in `outputs/`.

### Outputs (after running)
- `outputs/wordcloud.png`
- `outputs/tag_frequencies.csv`
- `outputs/network_nodes.csv`, `outputs/network_edges.csv`
- `outputs/network.png`
- `outputs/wordtree.html`
- `outputs/corpus.txt`, `outputs/infranodus_corpus.txt`

### Working code (minimal, reproduces outputs)
```python
import os, re, itertools
from collections import Counter
import pandas as pd, numpy as np

# Optional
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except Exception:
    WordCloud = None; plt = None
import networkx as nx
try:
    import community as community_louvain
except Exception:
    community_louvain = None

BASE = os.getcwd(); OUT = os.path.join(BASE, 'outputs'); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(BASE, 'agriculture_crop_yield.csv'))

fields = ['State','Crop_Type','Season','Climate_Zone','Soil_Type','Irrigation_Type','Pest_Infestation_Level','Disease_Incidence']
stop = set("a an the and or of in on for to with by from at as is are was were be been being low medium high summer winter continental subtropical mediterranean".split())
pat = re.compile(r"[a-zA-Z][a-zA-Z\-]+")

row_tokens = []
for i, r in df.iterrows():
    parts = [str(r[c]) for c in fields if c in df and pd.notna(r[c])]
    for col, p in [('Precipitation_mm','precip'),('Temperature_Celsius','temp'),('Fertilizer_Usage_kg','fert')]:
        if col in df and pd.notna(r.get(col)):
            b = pd.qcut(df[col], 3, labels=[f'{p}_low', f'{p}_med', f'{p}_high']).astype(str).iloc[i]
            parts.append(b)
    toks = [t.lower() for t in pat.findall(' '.join(parts)) if t.lower() not in stop]
    row_tokens.append(toks)

corpus = [' '.join(t) for t in row_tokens]
open(os.path.join(OUT,'corpus.txt'),'w').write('\n'.join(corpus))
open(os.path.join(OUT,'infranodus_corpus.txt'),'w').write('\n'.join(corpus))

allt = list(itertools.chain.from_iterable(row_tokens))
freq = Counter(allt)
pd.DataFrame(freq.most_common(), columns=['token','count']).to_csv(os.path.join(OUT,'tag_frequencies.csv'), index=False)

if WordCloud and plt:
    wc = WordCloud(width=1200, height=600, background_color='white').generate(' '.join(allt))
    plt.figure(figsize=(12,6)); plt.imshow(wc); plt.axis('off'); plt.savefig(os.path.join(OUT,'wordcloud.png'), bbox_inches='tight'); plt.close()

valid = {t for t,c in freq.items() if c>=3}
E = Counter()
for toks in row_tokens:
    s = sorted(set(t for t in toks if t in valid))
    for a,b in itertools.combinations(s,2):
        if a>b: a,b=b,a
        E[(a,b)] += 1
G = nx.Graph([(a,b,{'weight':w}) for (a,b),w in E.items()])
G.remove_nodes_from(list(nx.isolates(G)))

comm = {}
if community_louvain and G.number_of_nodes()>0:
    comm = community_louvain.best_partition(G, weight='weight')
else:
    for i,c in enumerate(nx.algorithms.community.greedy_modularity_communities(G, weight='weight')):
        for n in c: comm[n]=i

pd.DataFrame([
    {'id':n,'label':n,'degree':int(G.degree(n)),'freq':int(freq.get(n,0)),'community':int(comm.get(n,-1))}
    for n in G.nodes()
]).to_csv(os.path.join(OUT,'network_nodes.csv'), index=False)

pd.DataFrame([
    {'source':a,'target':b,'weight':int(d.get('weight',1))}
    for a,b,d in G.edges(data=True)
]).to_csv(os.path.join(OUT,'network_edges.csv'), index=False)

if plt and G.number_of_nodes()>0:
    pos = nx.spring_layout(G, seed=42, k=0.3)
    cs = [comm.get(n,-1) for n in G.nodes()]; u=sorted(set(cs)); cmap={c:plt.cm.tab20(i%20) for i,c in enumerate(u)}
    plt.figure(figsize=(12,8))
    nx.draw_networkx_nodes(G,pos,node_size=[300+20*freq.get(n,0) for n in G.nodes()],node_color=[cmap[c] for c in cs],alpha=.9)
    nx.draw_networkx_edges(G,pos,width=[.5+.3*d['weight'] for _,_,d in G.edges(data=True)],alpha=.4)
    nx.draw_networkx_labels(G,pos,{n:n for n in G.nodes()},font_size=8); plt.axis('off')
    plt.savefig(os.path.join(OUT,'network.png'), dpi=200, bbox_inches='tight'); plt.close()

root = max(freq.items(), key=lambda kv: kv[1])[0] if freq else None
if root:
    from html import escape
    paths=[]
    for line in corpus:
        t=line.split()
        for i,w in enumerate(t):
            if w==root:
                j=i; phrase=[]
                while j<len(t) and len(phrase)<4:
                    phrase.append(t[j]); j+=1
                paths.append(' '.join(phrase))
    with open(os.path.join(OUT,'wordtree.html'),'w') as f:
        f.write("""<!doctype html><html><head><meta charset='utf-8'><title>WordTree</title>
<script src='https://www.gstatic.com/charts/loader.js'></script><script>google.charts.load('current',{packages:['wordtree']});google.charts.setOnLoadCallback(draw);
function draw(){var d=google.visualization.arrayToDataTable([['Phrases'],
""")
        for p in paths: f.write("['"+escape(p)+"'],\n")
        f.write("""]);var o={wordtree:{format:'implicit',word:'"""+escape(root)+"""'}};var c=new google.visualization.WordTree(document.getElementById('wordtree'));c.draw(d,o);} </script></head><body><div id='wordtree' style='width:100%;height:700px;'></div></body></html>""")
print('Done ->', OUT)
```
