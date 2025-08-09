import torch, numpy as np
from torch.utils.data import Dataset
class ArielMAEDatasetV50(Dataset):
    def __init__(self, spectra: np.ndarray): self.spectra = spectra
    def __len__(self): return len(self.spectra)
    def __getitem__(self, i):
        mu = torch.tensor(self.spectra[i], dtype=torch.float32)
        mask = torch.ones_like(mu)
        return {"mu": mu, "mask_mu": mask}