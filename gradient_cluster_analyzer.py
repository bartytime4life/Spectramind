import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GradientClusterAnalyzer:
    def __init__(self, model, dataset, device="cuda", batch_size=32):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def extract_gradients(self):
        self.model.eval()
        grads, ids = [], []
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        for batch in tqdm(loader, desc="Extracting gradients"):
            fgs, airs, meta, planet_id = batch
            fgs, airs, meta = fgs.to(self.device), airs.to(self.device), meta.to(self.device)
            fgs.requires_grad_(True)
            self.model.zero_grad()
            mu, _ = self.model(fgs, airs, meta)
            mu.mean().backward()
            grad = fgs.grad.detach().mean(dim=(1, 2, 3)).cpu().numpy()
            grads.append(grad)
            ids.extend(planet_id)

        self.grad_matrix = np.concatenate(grads, axis=0)
        self.planet_ids = ids
        return self.grad_matrix

    def extract_latent(self):
        self.model.eval()
        z_all, ids = [], []
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        for batch in tqdm(loader, desc="Extracting latent vectors"):
            fgs, airs, meta, planet_id = batch
            fgs, airs, meta = fgs.to(self.device), airs.to(self.device), meta.to(self.device)
            with torch.no_grad():
                z = self.model.encode(fgs, airs, meta)  # Assumes model.encode() exists
            z_all.append(z.detach().cpu().numpy())
            ids.extend(planet_id)

        self.grad_matrix = np.concatenate(z_all, axis=0)
        self.planet_ids = ids
        return self.grad_matrix

    def cluster(self, algorithm="kmeans", n_clusters=5, eps=0.5, pca_components=2):
        reducer = PCA(n_components=pca_components)
        self.reduced = reducer.fit_transform(self.grad_matrix)

        if algorithm == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == "dbscan":
            clusterer = DBSCAN(eps=eps)
        elif algorithm == "gmm":
            clusterer = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        else:
            raise ValueError("Unknown algorithm: kmeans, dbscan, or gmm")

        self.labels = (
            clusterer.fit_predict(self.grad_matrix)
            if algorithm != "gmm"
            else clusterer.predict(self.grad_matrix)
        )

    def save_csv(self, out_csv="outputs/diagnostics/gradient_clusters.csv"):
        df = pd.DataFrame(self.reduced, columns=[f"PC{i+1}" for i in range(self.reduced.shape[1])])
        df["cluster"] = self.labels
        df["planet_id"] = self.planet_ids
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved gradient cluster file to {out_csv}")
        return df

    def attach_gll(self, df, labels_csv, submission_csv):
        label_df = pd.read_csv(labels_csv).set_index("planet_id")
        pred_df = pd.read_csv(submission_csv).set_index("planet_id")

        mu_cols = [f"mu_{i}" for i in range(283)]
        sigma_cols = [f"sigma_{i}" for i in range(283)]

        common = set(df["planet_id"]).intersection(label_df.index).intersection(pred_df.index)
        df = df.set_index("planet_id").loc[common]
        label_df = label_df.loc[common]
        pred_df = pred_df.loc[common]

        mu = pred_df[mu_cols].values
        sigma = np.clip(pred_df[sigma_cols].values, 1e-8, None)
        y_true = label_df[mu_cols].values

        gll = ((y_true - mu) / sigma) ** 2 + 2 * np.log(sigma)
        df["gll"] = gll.mean(axis=1)
        df.reset_index(inplace=True)

        gll_path = "outputs/diagnostics/gradient_clusters_with_gll.csv"
        df.to_csv(gll_path, index=False)

        summary = df.groupby("cluster").agg(
            count=("gll", "count"),
            mean_gll=("gll", "mean"),
            std_gll=("gll", "std"),
            median_gll=("gll", "median")
        )
        summary_path = gll_path.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path)
        print(f"📊 Cluster GLL summary saved to {summary_path}")

    def plot_pca_clusters(self, outdir="outputs/diagnostics", filename="gradient_cluster_plot.png"):
        os.makedirs(outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            self.reduced[:, 0], self.reduced[:, 1],
            c=self.labels, cmap="tab10", s=40, edgecolor="k"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Gradient Cluster Visualization (PCA)")
        legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize="small")
        ax.add_artist(legend)
        plot_path = os.path.join(outdir, filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"🖼️ Saved cluster PCA plot to {plot_path}")

    def analyze(self,
                algorithm="kmeans",
                n_clusters=5,
                labels_csv="data/train.csv",
                submission_csv="submission.csv",
                outdir="outputs/diagnostics",
                latent_mode=False):
        if latent_mode:
            print("⚙️ Extracting latent z vectors for clustering...")
            self.extract_latent()
        else:
            print("⚙️ Extracting input gradients for clustering...")
            self.extract_gradients()

        self.cluster(algorithm=algorithm, n_clusters=n_clusters)
        df = self.save_csv(os.path.join(outdir, "gradient_clusters.csv"))
        self.attach_gll(df, labels_csv, submission_csv)
        self.plot_pca_clusters(outdir=outdir)
