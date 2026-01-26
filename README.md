# DeepH-Immune  
**Hybrid Deep Learning Model for Identifying MHC Class II Immunogenic Epitopes Recognized by T Cell Receptors**

DeepH-Immune is a hybrid deep learning framework for predicting **MHC class II peptide immunogenicity** by explicitly modeling peptide–MHC interactions and integrating evolutionary information from protein language models.  
This repository provides **Python Jupyter notebooks** for training, evaluation, and interpretation of immunogenicity prediction models.

---

## Overview
Accurate identification of immunogenic peptide–MHC class II complexes is critical for cancer immunotherapy, vaccine development, and immune monitoring.  
DeepH-Immune integrates:

- Sequence-level convolutional encoders  
- Peptide–MHC **cross-attention** for residue-level interaction modeling  
- **ESM-2.0** protein language model embeddings for evolutionary context  
- Attention- and gradient-based interpretability  

to achieve accurate and biologically interpretable predictions.

---

## MHC Class II Data Requirements

Each input sample must include:

1. **HLA Allele Type**  
   - Format: `HLA-DRB1-0101`, `HLA-DRB5-0101`, etc.

2. **Peptide Sequence**  
   - Length: **13–15 amino acids**  
   - Example: `KAGVYKLTGAIMHYG`

3. **Immunogenicity Label**  
   - Binary label:  
     - `1` → Immunogenic  
     - `0` → Non-immunogenic  

---

### Example Input Data Format
    
Each row represents a peptide–MHC pair.

-HLA-DRB1-0101 KAGVYKLTGAIMHYG 0
-HLA-DRB5-0101 RFSWGAEGQRPGFGY 0

## Repository Structure

---

## Dependencies

Detailed execution instructions are provided within each Jupyter notebook.

### Key Dependencies
- TensorFlow (GPU-enabled recommended)
- Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## Utilized Versions
- **Python**: 3.11.14  
- **TensorFlow**: 2.20.0  
- **Keras**: 3.12.0  

> ⚠️ GPU acceleration (CUDA-enabled TensorFlow) is strongly recommended for training.

---

## Usage
1. Prepare input data following the specified format.
2. Open the appropriate Jupyter notebook in the `notebooks/` directory.
3. Configure file paths and hyperparameters as needed.
4. Run the notebook cells sequentially for training, evaluation, and analysis.

Each notebook contains inline comments explaining the workflow.

---

## Model Interpretability
DeepH-Immune supports multiple interpretability analyses, including:

- Peptide–MHC **cross-attention maps**
- Residue-level importance visualization
- **Integrated Gradients**–based motif discovery
- Per-allele MHC residue importance mapping

These methods provide biological insight into peptide recognition and MHC binding preferences.

---

## Applications
- Cancer neoantigen prioritization  
- Vaccine epitope discovery  
- Immunogenicity screening  
- T-cell epitope analysis  

---

## Contact
For questions, issues, or collaboration inquiries, please contact:

📧 **91979@ncc.re.kr**

---

## Citation
If you use DeepH-Immune in your research, please cite the corresponding publication (to be added upon acceptance/publication).

---

## License
This project is intended for academic and research use.  
Please contact the authors for licensing or commercial usage inquiries.

