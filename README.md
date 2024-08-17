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

Per lo sviluppo del lavoro progettuale sono state utilizzate 5 architetture basate su
Transformer per l’addestramento di modelli di Deep Learning per la classificazione di file audio. 
Le architetture considerate sono:
- Mockingjay
- Audio ALBERT
- Wav2vec 2.0
- HuBERT
- DistilHuBERT
