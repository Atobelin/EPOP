# Protein Language Modeling for Extracellular Matrix (ECM) Proteins

This repository contains code and data for protein sequence modeling using the **ESM-2** pre-trained model, with a focus on predicting and analyzing proteins related to the **extracellular matrix (ECM)**, particularly in **pelvic organ prolapse (POP)**.

## File Structure

- `notebooks/`: Contains all code scripts and notebooks
  - `01_data_preparation.ipynb`: Data preprocessing and partitioning
  - `02_esm2_training.py`: Fine-tuning ESM-2 model on ECM vs non-ECM sequences
  - `03_transformer_training.py`: Training Transformer-based model
  - `04_lstm_training.py`: Training LSTM-based model
  - `05_model_comparison_analysis.py`: Comparing model performance and plotting metrics
  - `06_pride.py`: PRIDE peptide prediction, deduplication, protein mapping, UniProt annotation
  - `07_get_epop_test_predictions_for_plot.py`: Inference on test set for final evaluation
- `data/`: Raw and processed input data
- `models/`: Saved models (ESM-2, Transformer, LSTM)
- `results/`: Intermediate outputs (pickled results, metrics, thresholds)
- `final_analysis/`: Final protein tables, evaluation plots, exportable figures
- `outputs/`: Training logs and other outputs
- `interpretability_plots/`: Optional attention or model interpretability visualizations

## Data Description

- `Extracellular_matrix_organization.tsv.gz`: ECM-related protein sequences used for model training/validation.
- `Not_extracellular_matrix_organization.tsv.gz`: Sequences not associated with ECM.
- `merged.results.F115714.mzid.gz`: PRIDE proteomic dataset (PXD011467), used for peptide-level prediction.
- `data_splits.pkl`: Contains stratified train/validation/test splits used across all models.

The `.tsv.gz` files should be extracted before use.

## Model Description

- **ESM-2**: Pretrained by FAIR (Meta AI), fine-tuned on ECM classification using a two-stage strategy (head → backbone).
- **Transformer**: Custom architecture implemented in Keras with multi-head attention and FFN blocks.
- **LSTM**: BiLSTM-based baseline sequence classifier with Keras.

Models are trained on stratified splits and evaluated on held-out test sets.

## Outputs Overview

- `esm2_enhanced_results.pkl`, `CustomTransformer_results.pkl`, `CustomLSTM_results.pkl`: Performance logs
- `epop_test_predictions_for_curves.pkl`: Test prediction scores for final curve drawing
- `model_performance.csv`: Final accuracy, F1, precision, recall metrics of all models
- `high_confidence_psms.csv`: Predicted high-scoring peptides from PRIDE dataset
- `deduplicated_high_confidence_psms.csv`: Deduplicated peptides and associated protein IDs
- `ensembl_uniprot_mapping.csv`: Mapping from Ensembl IDs to UniProt entries
- `peptides_with_uniprot_info.csv`: Peptides with UniProt annotations (gene name, protein name, reviewed status)
- Publication-ready plots:
  - `proteomics_analysis_combined.tif/png/eps`
  - `epop_curves.png`, `score_distributions.png`, `confusion_matrix.png`

##  Model Weights and Large Files

Due to file size limitations, some model weights and large intermediate files are hosted externally on **Google Drive**.

🔗 [Download from Google Drive](https://drive.google.com/drive/folders/18fi0rHCvyMTttGk6HTbPORsY1qaOTw5I?usp=drive_link)


Please download and manually place the files into corresponding subdirectories (e.g., `models/`, `results/`, or `final_analysis/`) according to their intended usage.

If you encounter any issues accessing or using these files, feel free to reach out.

##  References

- [ESM-2: Evolutionary-Scale Pre-Training](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)
- [PRIDE Project PXD011467](https://www.ebi.ac.uk/pride/archive/projects/PXD011467)
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [LSTM: Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

## Contact

Please contact us if you have any questions or would like to collaborate on ECM-related proteomic analysis.