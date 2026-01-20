import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataloader import ProteinStrainDataset
from model import HMIS
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from collections import defaultdict


protein_dir = "esm_data/basehit_protein"
strain_dir = "esm_data/strain"
stronghit_file = "strong_test.csv"
weakhit_file = "weak_test.csv"
negative_file = "negative_test.csv"
input_dim = 1280 * 2  
hidden_dim = 512
output_dim = 2  
batch_size = 16

dataset = ProteinStrainDataset(stronghit_file, weakhit_file, negative_file, protein_dir, strain_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = HMIS(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(f"S.pth"))
model.eval()  

all_outputs = []
all_probabilities = []  
true_labels = []
all_proteins = []
all_strains = []
all_sources = []

with torch.no_grad():
    for batch in dataloader:
        data = batch['data']
        labels = batch['label']
        proteins = batch['protein']
        strains = batch['strain']
        sources = batch['source']
        
        outputs = model(data)
        probabilities = torch.softmax(outputs, dim=1) 
        _, preds = torch.max(outputs, 1)
        
        all_outputs.extend(preds.cpu().numpy())
        all_probabilities.extend(probabilities[:, 1].cpu().numpy())  
        true_labels.extend(labels.cpu().numpy())
        all_proteins.extend(proteins)
        all_strains.extend(strains)
        all_sources.extend(sources)

results_df = pd.DataFrame({
    "protein": all_proteins,
    "strain": all_strains,
    "True_Label": true_labels,
    "Probability": all_probabilities, 
    "Predictions_0.5": all_outputs,  
    "Source": all_sources
})

thresholds = np.arange(0.1, 1.0, 0.1)
threshold_results = []

for threshold in thresholds:
 
    preds = (np.array(all_probabilities) >= threshold).astype(int)
    results_df[f"Predictions_{threshold:.1f}"] = preds  
    

    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, pos_label=1)
    recall = recall_score(true_labels, preds, pos_label=1)
    f1 = f1_score(true_labels, preds, pos_label=1)
    
    cm = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })

results_df.to_csv("final_predictions_with_thresholds.csv", index=False)
print("Predictions with multiple thresholds saved.")

threshold_metrics_df = pd.DataFrame(threshold_results)
threshold_metrics_df.to_csv("threshold_metrics.csv", index=False)
print("Threshold metrics saved.")

same_count = (results_df['True_Label'] == results_df['Predictions_0.5']).sum()
overall_accuracy = accuracy_score(true_labels, all_outputs)

overall_precision = precision_score(true_labels, all_outputs, pos_label=1)
overall_recall = recall_score(true_labels, all_outputs, pos_label=1)
overall_f1 = f1_score(true_labels, all_outputs, pos_label=1)

cm = confusion_matrix(true_labels, all_outputs)
tn, fp, fn, tp = cm.ravel()

print(f"\nOverall Results (0.5 threshold):")
print(f"Correct predictions: {same_count}/{len(results_df)} ({same_count/len(results_df):.2%})")
print(f"Accuracy: {overall_accuracy:.4f}")
print(f"Precision (positive class): {overall_precision:.4f}")
print(f"Recall (positive class): {overall_recall:.4f}")
print(f"F1 Score (positive class): {overall_f1:.4f}")
print("\nConfusion Matrix:")
print(f"True Negatives: {tn} | False Positives: {fp}")
print(f"False Negatives: {fn} | True Positives: {tp}")

source_metrics = defaultdict(dict)
unique_sources = results_df['Source'].unique()

for source in unique_sources:
    source_df = results_df[results_df['Source'] == source]
    if len(source_df) == 0:
        continue
        
    y_true = source_df['True_Label']
    y_pred = source_df['Predictions_0.5']
    
    source_metrics[source]['count'] = len(source_df)
    source_metrics[source]['accuracy'] = accuracy_score(y_true, y_pred)
    
    if len(set(y_true)) > 1:  
        source_metrics[source]['precision'] = precision_score(y_true, y_pred, pos_label=1)
        source_metrics[source]['recall'] = recall_score(y_true, y_pred, pos_label=1)
        source_metrics[source]['f1'] = f1_score(y_true, y_pred, pos_label=1)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        source_metrics[source]['tn'] = tn
        source_metrics[source]['fp'] = fp
        source_metrics[source]['fn'] = fn
        source_metrics[source]['tp'] = tp

print("\nMetrics by Source (0.5 threshold):")
for source, metrics in source_metrics.items():
    print(f"\nSource: {source}")
    print(f"  Samples: {metrics['count']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    if 'precision' in metrics:
        print(f"  Precision (positive): {metrics['precision']:.4f}")
        print(f"  Recall (positive): {metrics['recall']:.4f}")
        print(f"  F1 Score (positive): {metrics['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN: {metrics['tn']} | FP: {metrics['fp']}")
        print(f"    FN: {metrics['fn']} | TP: {metrics['tp']}")

metrics_df = pd.DataFrame.from_dict(source_metrics, orient='index')
metrics_df.to_csv("source_metrics.csv")
print("\nSource-specific metrics saved to source_metrics.csv")