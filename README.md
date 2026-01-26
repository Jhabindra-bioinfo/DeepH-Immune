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

## Model Interpretability
DeepH-Immune supports multiple interpretability analyses, including:

- Peptide–MHC **cross-attention maps**
- Residue-level importance visualization
- **Integrated Gradients**–based motif discovery
- Per-allele MHC residue importance mapping
- Allele clustering
These methods provide biological insight into peptide recognition and MHC binding preferences.


## Applications
- Cancer neoantigen prioritization   
- pMHC Immunogenicity prediction  
- Epitope motif
- Allele clustering
- Peptide-MHC interaction on the residue level


---

## MHC Class II Data Requirements

Each input sample must include:

1. **HLA (human) \H2 (mouse) Allele Type**  
   - `HLA-DRB1-0101`, `HLA-DRB5-0101`,  `H2-IAb`, etc.

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
---
### Key Dependencies
```bash
- TensorFlow (GPU-enabled recommended)
- Keras
- Torch
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
 ```


## Utilized Versions
```bash
- **Python**: 3.11.14  
- **TensorFlow**: 2.20.0  
- **Keras**: 3.12.0  
  ```

---

## Environment Setup 1
```bash
- conda create -n deeph-immune python=3.11.14 -y
- conda activate deeph-immune
- conda install -c conda-forge keras=3.12.0 tensorflow=2.20.0 -y
- conda install numpy scipy scikit-image -y
 ```


## Environment Setup 2

We also performed fine-tuning of **ESM-2.0 protein language models** for peptide immunogenicity prediction using **parameter-efficient fine-tuning (PEFT)** with **LoRA**.  
This enables effective adaptation of large pretrained protein models while keeping the number of trainable parameters small.

### Model

**Pretrained checkpoint:**
```bash
- facebook/esm2_t33_650M_UR50D (link: https://huggingface.co/facebook/esm2_t33_650M_UR50D)
  ```
### Core Libraries
```bash
- PyTorch
```

### Hugging Face Transformers
 ```bash
  - AutoTokenizer
  - EsmForSequenceClassification
  - Trainer, TrainingArguments
  - EarlyStoppingCallback
  - PEFT (LoRA)
  - get_peft_model
  - LoraConfig
  - PeftModel
```
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



