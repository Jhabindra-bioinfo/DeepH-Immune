# DeepH-Immune  
**Hybrid Deep Learning Model for Identifying MHC Class II Immunogenic Epitopes Recognized by T Cell Receptors**

DeepH-Immune is a hybrid deep learning framework for predicting **MHC class II peptide immunogenicity** by explicitly modeling peptide–MHC interactions and integrating evolutionary information from protein language models.  
This repository provides **Python Jupyter notebooks** for training, evaluation, and interpretation.

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

1. **HLA (human) \H2 (mouse) Allele Type**  
   - Format: `HLA-DRB1-0101`, `HLA-DRB5-0101`, 'H2-IAb', etc.

2. **Peptide Sequence**  
   - Length: **13–15 amino acids**  
   - Example: `KAGVYKLTGAIMHYG`

3. **Immunogenicity Label**  
   - Binary label:  
     - `1` → Immunogenic  
     - `0` → Non-immunogenic  


### Example Input Data Format
    Each row represents a peptide–MHC pair with immunogenicity labels (o or 1).
    -HLA-DRB1-0101 KAGVYKLTGAIMHYG 0
    -HLA-DRB5-0101 RFSWGAEGQRPGFGY 0

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

---

## Model Interpretability
DeepH-Immune supports multiple interpretability analyses, including:

- Peptide–MHC **cross-attention maps**
- Residue-level importance visualization
- **Integrated Gradients**–based motif discovery
- Per-allele MHC residue importance mapping
- Allele clustering
These methods provide biological insight into peptide recognition and MHC binding preferences.

---

## Applications
- Cancer neoantigen prioritization   
- pMHC Immunogenicity prediction  
- Epitope motif
- Allele clustering
- Peptide-MHC interaction on the residue level

---

## Contact
For questions, issues, or collaboration inquiries, please contact:

📧 **91979@ncc.re.kr**

## Workflow

<img width="398" height="193" alt="image" src="https://github.com/user-attachments/assets/f0b81e24-5e61-43d5-bf06-19a419b0d7f0" />

**Figure:** Overview of the DeepH-Immune architecture. 

---
## License
This project is intended for academic and research use.  



