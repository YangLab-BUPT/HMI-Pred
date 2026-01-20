import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class ProteinStrainDataset(Dataset):
    def __init__(self, stronghit_file, weakhit_file, negative_file, protein_dir, strain_dir, 
                 
                 max_samples_per_class=100000):
 
        self.protein_dir = protein_dir
        self.strain_dir = strain_dir
        self.data = []
        self.labels = []
        self.proteins = []
        self.strains = []
        self.sources = []  

        self._load_data(stronghit_file, 0, max_samples_per_class, source='nature')
 
        self._load_data(weakhit_file, 0, max_samples_per_class, source='nature')
  
        self._load_data(negative_file, 1, max_samples_per_class, source='nature')
        
        

        
        self._print_class_counts()

    def _load_data(self, file_path, label, max_samples, source):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        sample_count = 0

        with open(file_path, 'r') as file:
            for line in file:
                if sample_count >= max_samples:
                    break

                protein, strain = line.strip().split(',')[:2]
                protein_path = os.path.join(self.protein_dir, f"{protein}.pt")
                strain_path = os.path.join(self.strain_dir, strain)  
                
                if os.path.exists(protein_path) and os.path.exists(strain_path):
                    protein_data = torch.load(protein_path)
                    strain_files = [f for f in os.listdir(strain_path) if f.endswith('.pt')][:2]
                    
                    for strain_file in strain_files:
                        strain_data = torch.load(os.path.join(strain_path, strain_file))
                        strain_vector = list(strain_data['mean_representations'].values())[0]
                        protein_vector = list(protein_data['mean_representations'].values())[0]

                        if torch.isnan(protein_vector).any() or torch.isinf(protein_vector).any():
                            print(f"NaN or Inf found in protein vector for {protein}")
                            continue
                        if torch.isnan(strain_vector).any() or torch.isinf(strain_vector).any():
                            print(f"NaN or Inf found in strain vectors for {strain}")
                            continue

                        combined_vector = torch.cat((protein_vector, strain_vector), dim=0)
                        self.proteins.append(protein)
                        self.strains.append(strain)
                        self.data.append(combined_vector)
                        self.labels.append(label)
                        self.sources.append(source) 

                sample_count += 1

                
    def _print_class_counts(self):
        unique_labels, counts = torch.unique(torch.tensor(self.labels), return_counts=True)
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"Class {label}: {count} samples")

    def get_class_counts(self):
        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[label] += 1
        return class_counts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'protein': self.proteins[idx],
            'strain': self.strains[idx],
            'source': self.sources[idx] 
        }