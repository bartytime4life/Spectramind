python plot_umap_v50.py \
  --config configs/config_v50.yaml \
  --checkpoint outputs/model.pt \
  --overlay_csv diagnostics/symbolic_clusters.csv \
  --overlay_column symbolic_class \
  --confidence_column confidence \
  --link_template https://planetdash.io/{planet_id}