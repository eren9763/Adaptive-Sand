# External SAND package

Bu klasöre orijinal SAND kodunu koymalısın.

Bu proje şu import yollarını bekler:

```python
from sand.experiments.datasets.dataset import get_dataset
from sand.experiments.models.mlp_sand import SANDModel
from sand.experiments.models.mlp_sa import SequentialAttentionModel
```

Eğer SAND kodun ayrı bir repoda ise buraya submodule olarak ekleyebilirsin:

```bash
git submodule add <sand-repo-url> sand
```

