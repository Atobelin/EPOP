# Protein Language Modeling for Extracellular Matrix (ECM) Proteins

This repository contains code and data for protein sequence modeling using the ESM-2 pre-trained model, with a focus on predicting and analyzing proteins related to the Extracellular Matrix (ECM).

## File Structure

- `protein_language_modeling_ecm_tf.ipynb`: Code file for training and validating the model using the ESM-2 pre-trained model, implemented using the TensorFlow framework.
- `pride.ipynb`: Code file for retrieving and validating proteomics data, and using our trained model to predict peptides in the proteomics data to find ECM-related proteins. Also includes model interpretability analysis.
- `ecm_comparison.ipynb`: Code file for training models using Transformer and LSTM on the same data for performance comparison.
- `Extracellular_matrix_organization.tsv.gz`: Protein sequence data file related to ECM.
- `Not_extracellular_matrix_organization.tsv.gz`: Protein sequence data file not related to ECM.
- `esm_ecm/`: Folder for saving trained model files.

## Usage Instructions

1. Ensure that you have installed the necessary Python libraries, such as TensorFlow and Jupyter Notebook.
2. Download and extract the code repository to your local machine.
3. Run the `protein_language_modeling_ecm_tf.ipynb` file to train and validate the ESM-2 model.
4. Run the `pride.ipynb` file to predict and analyze proteomics data using the trained model.
5. (Optional) Run the `ecm_comparison.ipynb` file to train and compare models using Transformer and LSTM on the same data.

## Data Description

- `Extracellular_matrix_organization.tsv.gz`: Contains protein sequences related to ECM, used for model training and validation.
- `Not_extracellular_matrix_organization.tsv.gz`: Contains protein sequences not related to ECM, used for model training and validation.

The data files are in compressed format and need to be extracted before use.

## Model Description

- ESM-2: Uses the ESM-2 pre-trained model provided by Facebook AI Research (FAIR) for protein sequence modeling.
- Transformer: Uses the Transformer model for protein sequence modeling, for performance comparison with the ESM-2 model.
- LSTM: Uses the Long Short-Term Memory (LSTM) model for protein sequence modeling, for performance comparison with the ESM-2 model.

Trained model files are saved in the `esm_ecm/` folder.

## References

- [ESM-2: Evolutionary-Scale Pre-Training for Protein Language Models](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)
- [Transformer: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LSTM: Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

Please feel free to contact us if you have any questions.
