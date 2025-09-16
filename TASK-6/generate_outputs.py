#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
	import networkx as nx
	HAS_NX = True
except Exception:
	HAS_NX = False


def main():
	data_path = Path('/Users/karthikmac/Downloads/DV_USECASE/task 6/agriculture_crop_yield.csv')
	out_dir = Path('/Users/karthikmac/Downloads/DV_USECASE/task 6/outputs')
	out_dir.mkdir(parents=True, exist_ok=True)

	# Style
	sns.set_theme(style='whitegrid')
	plt.rcParams['figure.figsize'] = (10, 6)

	# Load
	df = pd.read_csv(data_path)
	if 'Harvest_Date' in df.columns:
		df['Harvest_Date'] = pd.to_datetime(df['Harvest_Date'], errors='coerce')
	if {'Yield_Tonnes','Area_Hectares'}.issubset(df.columns):
		df['Yield_per_Hectare_calc'] = df['Yield_Tonnes'] / df['Area_Hectares']

	# ---------- Graphs ----------
	# 1) Time series per state (subset) by crop
	pivot = df.groupby(['Year','State','Crop_Type'], as_index=False)['Yield_Tonnes'].sum()
	states = sorted(pivot['State'].unique())[:6]
	fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
	axes = axes.flatten()
	for ax, state in zip(axes.flatten(), states):
		state_df = pivot[pivot['State'] == state]
		sns.lineplot(data=state_df, x='Year', y='Yield_Tonnes', hue='Crop_Type', marker='o', ax=ax, legend=False)
		ax.set_title(state)
		ax.set_ylabel('Total Yield (Tonnes)')
	fig.tight_layout()
	fig.savefig(out_dir / 'timeseries_yield_by_state_crop.png', dpi=200)
	plt.close(fig)

	# 2) Bar: total yield by crop
	crop_totals = df.groupby('Crop_Type', as_index=False)['Yield_Tonnes'].sum().sort_values('Yield_Tonnes', ascending=False)
	plt.figure(figsize=(10, 6))
	sns.barplot(data=crop_totals, x='Yield_Tonnes', y='Crop_Type', palette='viridis')
	plt.title('Total Yield by Crop Type')
	plt.xlabel('Yield (Tonnes)')
	plt.ylabel('Crop Type')
	plt.tight_layout()
	plt.savefig(out_dir / 'total_yield_by_crop.png', dpi=200)
	plt.close()

	# 3) Heatmap: Yield per hectare state x crop (if available)
	if 'Yield_per_Hectare' in df.columns:
		heat = df.groupby(['State','Crop_Type'])['Yield_per_Hectare'].mean().unstack()
		plt.figure(figsize=(12, max(6, 0.35 * len(heat.index))))
		sns.heatmap(heat, cmap='YlGnBu', linewidths=.5)
		plt.title('Average Yield per Hectare by State and Crop')
		plt.xlabel('Crop Type')
		plt.ylabel('State')
		plt.tight_layout()
		plt.savefig(out_dir / 'yield_per_hectare_heatmap.png', dpi=200)
		plt.close()

	# 4) Scatter: precipitation vs yield per hectare
	if {'Precipitation_mm','Yield_per_Hectare'}.issubset(df.columns):
		plt.figure(figsize=(10, 6))
		sns.scatterplot(data=df, x='Precipitation_mm', y='Yield_per_Hectare', hue='Crop_Type', alpha=0.7)
		plt.title('Precipitation vs Yield per Hectare')
		plt.tight_layout()
		plt.savefig(out_dir / 'precip_vs_yield_per_hectare_scatter.png', dpi=200)
		plt.close()

	# ---------- Networks ----------
	if HAS_NX:
		# A) Bipartite State–Crop network (force-directed)
		B = nx.Graph()
		states_all = sorted(df['State'].unique())
		crops_all = sorted(df['Crop_Type'].unique())
		B.add_nodes_from(states_all, bipartite=0, kind='state')
		B.add_nodes_from(crops_all, bipartite=1, kind='crop')
		edges = (
			df.groupby(['State','Crop_Type'])['Yield_Tonnes']
			.sum()
			.reset_index()
		)
		for _, row in edges.iterrows():
			B.add_edge(row['State'], row['Crop_Type'], weight=float(row['Yield_Tonnes']))
		pos = nx.spring_layout(B, seed=42, weight='weight')
		plt.figure(figsize=(14, 10))
		max_w = float(edges['Yield_Tonnes'].max() or 1.0)
		edge_w = [1 + 3*(B[u][v]['weight'] / max_w) for u, v in B.edges()]
		nx.draw_networkx_edges(B, pos, alpha=0.25, width=edge_w)
		nx.draw_networkx_nodes(B, pos, nodelist=states_all, node_color='#1f77b4', node_size=400, label='States')
		nx.draw_networkx_nodes(B, pos, nodelist=crops_all, node_color='#ff7f0e', node_shape='s', node_size=400, label='Crops')
		nx.draw_networkx_labels(B, pos, font_size=8)
		plt.title('Bipartite Network (States–Crops), force-directed')
		plt.axis('off')
		plt.legend(scatterpoints=1)
		plt.tight_layout()
		plt.savefig(out_dir / 'network_bipartite_states_crops.png', dpi=200)
		plt.close()

		# B) Numeric feature correlation network (force-directed)
		numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		corr = df[numeric_cols].corr()
		threshold = 0.6
		G_corr = nx.Graph()
		G_corr.add_nodes_from(numeric_cols)
		for i, a in enumerate(numeric_cols):
			for j in range(i+1, len(numeric_cols)):
				b = numeric_cols[j]
				val = float(corr.loc[a, b])
				if abs(val) >= threshold:
					G_corr.add_edge(a, b, weight=abs(val), sign='pos' if val >= 0 else 'neg')
		pos = nx.spring_layout(G_corr, seed=7, weight='weight')
		plt.figure(figsize=(11, 9))
		edge_colors = ['#2ca02c' if G_corr[u][v]['sign']=='pos' else '#d62728' for u, v in G_corr.edges()]
		edge_w = [2 + 4*G_corr[u][v]['weight'] for u, v in G_corr.edges()]
		nx.draw_networkx_edges(G_corr, pos, edge_color=edge_colors, width=edge_w, alpha=0.5)
		nx.draw_networkx_nodes(G_corr, pos, node_color='#1f77b4', node_size=800)
		nx.draw_networkx_labels(G_corr, pos, font_size=9, font_color='white')
		plt.title(f'Feature Correlation Network (|r| ≥ {threshold}), force-directed')
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(out_dir / 'network_feature_correlation.png', dpi=200)
		plt.close()

		# C) Transportation-like proxy network + hubs (force-directed)
		state_crop = df.groupby(['State','Crop_Type'], as_index=False)['Yield_Tonnes'].sum()
		M = state_crop.pivot(index='State', columns='Crop_Type', values='Yield_Tonnes').fillna(0)
		states_idx = M.index.tolist()
		from numpy.linalg import norm
		def cosine(a, b):
			na, nb = norm(a), norm(b)
			if na == 0 or nb == 0:
				return 0.0
			return float(np.dot(a, b) / (na * nb))
		sim_threshold = 0.6
		G_trans = nx.Graph()
		G_trans.add_nodes_from(states_idx)
		for i in range(len(states_idx)):
			for j in range(i+1, len(states_idx)):
				s = cosine(M.iloc[i].values, M.iloc[j].values)
				if s >= sim_threshold:
					G_trans.add_edge(states_idx[i], states_idx[j], weight=s)
		# Centralities
		deg = nx.degree_centrality(G_trans)
		bet = nx.betweenness_centrality(G_trans, weight='weight', normalized=True)
		try:
			eig = nx.eigenvector_centrality(G_trans, weight='weight', max_iter=2000)
		except Exception:
			eig = {n: np.nan for n in G_trans.nodes}
		# Save hub metrics
		hubs_df = pd.DataFrame({
			'state': list(G_trans.nodes()),
			'degree_centrality': [deg.get(n, np.nan) for n in G_trans.nodes()],
			'betweenness_centrality': [bet.get(n, np.nan) for n in G_trans.nodes()],
			'eigenvector_centrality': [eig.get(n, np.nan) for n in G_trans.nodes()],
		}).sort_values('betweenness_centrality', ascending=False)
		hubs_df.to_csv(out_dir / 'transport_hub_metrics.csv', index=False)
		# Visualization
		pos = nx.spring_layout(G_trans, seed=13, weight='weight')
		node_sizes = np.array([200 + 1800*deg.get(n, 0.0) for n in G_trans.nodes()])
		node_colors = np.array([bet.get(n, 0.0) for n in G_trans.nodes()])
		plt.figure(figsize=(12, 9))
		nx.draw_networkx_edges(G_trans, pos, alpha=0.3, width=[1 + 3*G_trans[u][v]['weight'] for u,v in G_trans.edges()])
		nodes = nx.draw_networkx_nodes(G_trans, pos, node_size=node_sizes, node_color=node_colors, cmap='plasma')
		nx.draw_networkx_labels(G_trans, pos, font_size=9)
		plt.colorbar(nodes, label='Betweenness centrality')
		plt.title('Transportation Proxy Network (force-directed). Size~Degree, Color~Betweenness')
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(out_dir / 'network_transport_proxy.png', dpi=200)
		plt.close()

		# Also export a simplified layout styled like the sample (lightgreen nodes)
		plt.figure(figsize=(12, 8))
		nx.draw(
			G_trans,
			pos=pos,
			with_labels=True,
			node_size=1000,
			node_color='lightgreen',
			font_size=10,
			edge_color='#888',
			width=1.5,
		)
		plt.title('Transportation Network - Force-Based Layout')
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(out_dir / 'network_transport_force_simple.png', dpi=200)
		plt.close()

		# Build a single-page PDF-like figure with Aim/Algorithm/Program/Output/Result
		try:
			import matplotlib.image as mpimg
			img_path = out_dir / 'network_transport_force_simple.png'
			img = mpimg.imread(img_path)
			fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
			# Title
			fig.suptitle('dv lab manual', fontsize=10, y=0.99)
			# Aim
			ax_aim = fig.add_axes([0.07, 0.93, 0.86, 0.04])
			ax_aim.axis('off')
			ax_aim.text(0, 0.5, 'Aim: To design and perform visualization for Graphs and Networks', fontsize=10, va='center')
			# Algorithm
			ax_alg = fig.add_axes([0.07, 0.86, 0.86, 0.06])
			ax_alg.axis('off')
			algorithm_lines = [
				'Algorithm:',
				'1. Load the dataset',
				'2. Import modules (networkx as nx, pandas, matplotlib)',
				'3. Construct transportation graph (nodes=states, weighted edges by similarity)',
				'4. Apply force-based layout (spring)',
				'5. Visualize and label nodes',
				'6. Interpret hubs/insights',
			]
			ax_alg.text(0, 1, '\n'.join(algorithm_lines), fontsize=9, va='top')
			# Program (snippet)
			ax_prog = fig.add_axes([0.07, 0.72, 0.86, 0.1])
			ax_prog.axis('off')
			program_text = (
				'Program:\n'
				'pos = nx.spring_layout(G_trans, seed=13, weight=\'weight\')\n'
				"plt.figure(figsize=(12, 8))\n"
				"nx.draw(G_trans, pos, with_labels=True, node_size=1000, node_color='lightgreen', font_size=10)\n"
				"plt.title('Transportation Network - Force-Based Layout')\nplt.show()"
			)
			ax_prog.text(0, 1, program_text, fontsize=9, va='top', family='monospace')
			# Output image
			ax_img = fig.add_axes([0.1, 0.37, 0.8, 0.32])
			ax_img.imshow(img)
			ax_img.axis('off')
			ax_lbl = fig.add_axes([0.1, 0.32, 0.8, 0.04])
			ax_lbl.axis('off')
			ax_lbl.text(0, 0.5, 'Output:', fontsize=11, va='center')
			# Result
			ax_res = fig.add_axes([0.05, 0.06, 0.9, 0.2])
			ax_res.axis('off')
			ax_res.text(
				0.0,
				0.5,
				'Thus design and perform visualization for Graphs and Networks successfully completed',
				fontsize=11,
				va='center',
			)
			fig.savefig(out_dir / 'dv_report.pdf')
			plt.close(fig)
		except Exception:
			pass

		# Cleanup: keep only the simple network image and the PDF report
		for p in out_dir.iterdir():
			if p.name not in {'network_transport_force_simple.png', 'dv_report.pdf'}:
				try:
					p.unlink()
				except Exception:
					pass
		# Export graph for Gephi
		try:
			nx.write_gexf(G_trans, out_dir / 'transport_proxy_network.gexf')
		except Exception:
			pass

	print(f"Outputs written to: {out_dir}")


if __name__ == '__main__':
	main()


