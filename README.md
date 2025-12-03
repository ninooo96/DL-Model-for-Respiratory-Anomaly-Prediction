# DL-Model-for-Respiratory-Anomaly-Prediction
## Progetto di tesi Magistrale in Ingegneria Informatica (indirizzo Artificial Intelligence and Machine Learning)

Lo scopo di questo lavoro di tesi è quello di studiare le moderne architetture di Deep Learning di tipo Transformer e l'addestramento di modelli per la classificazione di anomalie respiratorie attraverso l’analisi di file audio di auscultazione polmonare.

----------------------
Il notebook “Transformers - ICBHI dataset.ipynb” è stato utilizzato per effettuare tutte le operazioni di download, creazione e salvataggio dei dataset utilizzati, non solo ICBHI. Sono, inoltre, contenuti tutti gli esperimenti di preprocessing eseguiti sui dataset.

Il notebook “Data preprocessing.ipynb” è stato utilizzato per effettuare tutte le operazioni di preprocessing sui dataset:
- Creazione dataset cicli respiratori
- Creazione dataset cicli respiratori con tecnica di data augmentation
- Applicazione filtro passa-banda su HF_Lung_V1
- Applicazione filtro passa-banda su ICBHI anomaly-driven
- Applicazione filtro passa-banda su ICBHI pathology-driven

Il notebook “Models evaluation.ipynb” è stato utilizzato per il caricamento e la valutazione (sui test set) delle versioni che hanno ottenuto le performance migliori per ogni modello addestrato.

Il notebook “Dataset HF_Lung_V1.ipynb” è stato utilizzato per il download e la creazione della versione finale del dataset HF_Lung_V1 solo per il caricamento su Kaggle

Per lo sviluppo del lavoro progettuale sono state utilizzate **5 architetture basate su Transformer** per l’addestramento di modelli di Deep Learning per la classificazione di file audio. 
Le architetture considerate sono:
- Mockingjay
- Audio ALBERT
- Wav2vec 2.0
- HuBERT
- DistilHuBERT

Vi è una cartella dedicata per **ogni modello Transformer addestrato**, all'interno di ciascuna sono presenti i notebook utili all'addestramento (fine-tuning) del modello per i due task principali:

* **Task 1: Anomaly-Driven Prediction** (Classificazione di cicli respiratori: rilevamento di Crepitii/Rantoli).
* **Task 2: Pathology-Driven Prediction** (Classificazione dell'intera clip audio per la diagnosi della patologia).

---

## Risultati Sperimentali

La performance è stata valutata utilizzando l'**ICBHI Score** ($\frac{Sensitivity + Specificity}{2}$). I modelli sono stati testati su due task principali.

### 1. Anomaly-Driven Prediction (Classificazione di Cicli Respiratori)
*Obiettivo: Rilevare anomalie (Crepitii/Rantoli) nei singoli cicli respiratori.*

#### Classificazione Binaria (Healthy vs Unhealthy)

| Modello | Sensitivity (Se) | Specificity (Sp) | Accuracy | ICBHI Score |
| :--- | :---: | :---: | :---: | :---: |
| Mockingjay | 0.72 | 0.72 | 0.72 | 0.72 |
| Audio ALBERT | 0.53 | **0.88** | 0.71 | 0.70 |
| Wav2vec 2.0 | 0.66 | 0.81 | 0.73 | 0.72 |
| HuBERT | **0.78** | 0.73 | 0.75 | 0.75 |
| **DistilHuBERT** 🏆 | 0.69 | 0.83 | **0.76** | **0.76** |
| DistilHuBERT (HF Lung V1) | 0.66 | 0.82 | 0.74 | 0.74 |

#### Classificazione Multiclasse (Crackle, Wheeze, Both, Healthy)

| Modello | Sensitivity (Se) | Specificity (Sp) | Accuracy | ICBHI Score |
| :--- | :---: | :---: | :---: | :---: |
| Mockingjay | 0.43 | 0.84 | 0.63 | 0.63 |
| Audio ALBERT | 0.43 | **0.89** | 0.66 | 0.66 |
| Wav2vec 2.0 | 0.47 | 0.86 | 0.66 | 0.66 |
| HuBERT | 0.50 | 0.85 | 0.68 | 0.68 |
| **DistilHuBERT** 🏆 | **0.56** | 0.81 | **0.69** | **0.69** |
| DistilHuBERT (HF Lung V1) | 0.44 | **0.89** | 0.67 | 0.66 |

### 2. Pathology-Driven Prediction (Diagnosi della Patologia)
*Obiettivo: Diagnosticare la patologia del paziente basandosi sull'intera registrazione.*

#### Classificazione Binaria (Sano vs Malato)

| Modello | Sensitivity (Se) | Specificity (Sp) | Accuracy | ICBHI Score |
| :--- | :---: | :---: | :---: | :---: |
| Mockingjay | **1.00** | 0.00 | 0.95 | 0.50 |
| Audio ALBERT | 0.99 | 0.20 | 0.95 | 0.60 |
| Wav2vec 2.0 | **1.00** | 0.00 | 0.95 | 0.50 |
| HuBERT | **1.00** | 0.00 | 0.95 | 0.50 |
| **DistilHuBERT** 🏆 | 0.98 | **0.50** | **0.96** | **0.74** |
| **DistilHuBERT (HF Lung V1)** 🏆 | 0.99 | **0.50** | **0.96** | **0.74** |

#### Classificazione Multiclasse (Chronic, Non-Chronic, Healthy)

| Modello | Sensitivity (Se) | Specificity (Sp) | Accuracy | ICBHI Score |
| :--- | :---: | :---: | :---: | :---: |
| Mockingjay | 0.94 | 0.00 | 0.89 | 0.47 |
| **Audio ALBERT** 🏆 | 0.97 | **0.50** | 0.94 | **0.73** |
| Wav2vec 2.0 | 0.95 | 0.00 | 0.90 | 0.47 |
| HuBERT | **0.98** | 0.00 | 0.92 | 0.49 |
| DistilHuBERT | 0.96 | 0.00 | 0.91 | 0.48 |
| **DistilHuBERT (HF Lung V1)** 🏆 | 0.97 | **0.50** | **0.95** | **0.73** |

---

## Conclusioni

* **Efficacia dei Transformer:** Dimostrata l'eccellente capacità di *transfer learning* dei modelli Transformer (pre-addestrati su parlato) applicati a domini completamente diversi come i suoni respiratori.
* **Best Performer:** L'architettura **DistilHuBERT** ha offerto il miglior trade-off tra performance (alto ICBHI Score) e leggerezza computazionale.
* **Metodologia:** Il lavoro conferma che le architetture Transformer con feature extraction interna (input Raw Waveform) sono il metodo più efficace per questo tipo di classificazione.

