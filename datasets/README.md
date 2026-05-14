# Datasets

Dataset dosyalarını bu klasör altında tutabilirsin.

Önerilen yapı:

```text
datasets/
├── raw/
│   ├── isolet/
│   ├── coil/
│   ├── activity/
│   └── mice/
└── processed/
```

Mevcut SAND dataset loader hangi dosya yollarını bekliyorsa aynı yapıyı korumalısın.

Alternatif olarak ortam değişkeni kullanabilirsin:

```bash
export SAND_DATA_DIR=/path/to/datasets
python main.py --dataset isolet --model hybrid
```

