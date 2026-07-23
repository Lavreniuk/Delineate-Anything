# [ECCV 2026] Delineate Anything v2: A Global Foundation Model for Field Delineation
<a href='https://lavreniuk.github.io/Delineate-Anything/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2504.02534'><img src='https://img.shields.io/badge/Paper-DelAny-red'></a>
<a href='https://arxiv.org/abs/2511.13417'><img src='https://img.shields.io/badge/Paper-DelAnyFlow-red'></a>
<a href='https://arxiv.org/abs/2607.19069'><img src='https://img.shields.io/badge/Paper-DelAny v2-red'></a>
<a href='https://explorer.delineate-anything.apex.esa.int/'><img src='https://img.shields.io/badge/Map-Explorer-blue'></a>
<a href='https://huggingface.co/datasets/MykolaL/FBIS-22M'><img src='https://img.shields.io/badge/Dataset-HuggingFace-DA0000'></a>
<a href='https://colab.research.google.com/drive/10KSLwYDTgU-WhpqqG39yyvB6K8MdB0X9?usp=sharing'><img src='https://img.shields.io/badge/Colab-Demo-F9AB00'></a>

<p align="center">
  <img src="figs/logo.jpg" alt="intro" width="448"/>
</p>


by [Mykola Lavreniuk](https://scholar.google.com/citations?hl=en&user=-oFR-RYAAAAJ), [Nataliia Kussul](https://scholar.google.com/citations?user=e3TWBuwAAAAJ&hl=en), [Andrii Shelestov](https://scholar.google.com/citations?user=tqoQKZAAAAAJ&hl=en), [Bohdan Yailymov](https://scholar.google.com/citations?user=XaN-oukAAAAJ&hl=en), [Yevhenii Salii](https://scholar.google.com/citations?user=4jgAsBIAAAAJ&hl=en), [Volodymyr Kuzin](https://www.researchgate.net/profile/Volodymyr-Kuzin), [Zoltan Szantoi](https://scholar.google.com/citations?user=P_pyhi8AAAAJ&hl=en)

**Delineate Anything v2** extends Delineate Anything model into a globally representative, resolution-agnostic foundation model that scales agricultural field boundary detection to a planetary level from any imagery source. Trained on **FBIS-73M**, a massive 73-million-instance dataset spanning 61 countries with diverse imagery sources ranging from 0.25m to 10m resolution, built through a resolution-specific curation pipeline that solves the parcel-versus-field mismatch, Delineate Anything v2 sets a new state-of-the-art in global zero-shot delineation. It delivers a +103.3% relative gain in mAP@0.5 over Delineate Anything while maintaining extreme efficiency, mapping all of Ukraine ($603,000 \text{ km}^2$) in 5.4 hours on a regular PC with 1 GPU NVIDIA RTX 5070 Ti 16 GB.

![intro](figs/intro.jpg)
![intro](figs/intro_v2.jpg)


## News
- `2026/07/16`: 🔥 **[Delineate Anything v2: A Global Foundation Model for Field Delineation](https://arxiv.org/abs/2607.19069)** is accepted to **ECCV 2026**!
- `2025/11/17`: **[Delineate Anything Flow: Fast, Country-Level Field Boundary Detection from Any Source](https://arxiv.org/abs/2511.13417)** published!
- `2025/09/07`: 🚀🚀🚀 [Autobounds](https://autobounds.com/) released for convenient **field boundary detection** with Delineate-Anything, directly in the browser!  
   👉 [Demo Video](http://bit.ly/4ngrM9k) | [Live App](https://autobounds.com/ai-models).
- `2025/08/30`: 🚀🚀 Our paper on Delineate-Anything accepted at **ECAI 2025** 🎉.
- `2025/07/07`: 🚀 Delineate-Anything integrated into the [TorchGeo library](https://huggingface.co/torchgeo/delineate-anything/blob/main/README.md).

## 📊 Models & Performance

### Global Benchmark (100-Country Independent Evaluation)
| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency (ms) | Size | Download |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Delineate Anything | 0.275 | 0.103 | 0.345 | 0.454 | 25.0 | 125 MB | [Download](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt?download=true) |
| **Delineate Anything v2** | **0.559** | **0.278** | **0.639** | **0.525** | 25.0 | 125 MB | [Download](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnythingv2.pt?download=true) |

### Regional Performance Breakdown (mAP@0.5)
| Method | Europe | Africa | Asia & Oceania | Latin America | North America |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Delineate Anything | 0.332 | 0.251 | 0.161 | 0.314 | 0.317 |
| **Delineate Anything v2** | **0.612** | **0.584** | **0.440** | **0.563** | **0.618** |

*\*Note: Both models in the global benchmark are evaluated on the newly curated manual benchmark spanning 100 countries! Original baseline evaluation from the Delineate Anything paper used FBIS-22M test splits, which primarily covered only Europe.*

## ⚙️ Environment Setup

To set up the environment on a Linux system:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda install -c conda-forge gdal

# optional: pip install torch==2.6.0
pip install -r requirements.txt
```

To set up the environment on a Windows system:

```bash
conda create --prefix=./.conda python=3.11
conda activate ./.conda
conda install -c conda-forge gdal
# optional: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```



## 🚀 Inference
💡 Try the Colab demo first, **no installation needed**, or run locally if you prefer full control.

1. Place your RGB images in the `data/images/` folder. If available, also include the corresponding land cover map in the `data/masks/`
   _(Three Sentinel-2 sample images and a land cover map are provided for testing.)_

2. Run the inference script:

   ```bash
   python delineate.py -b batch_sample.yaml
   ```
   
   The vectorized field boundaries will be saved as a GeoPackage in:
   ```data/delineated/```

3. (Optional) To shift the resulting vector geometries:
   
   Shift using image pixels:
   ```
   python shift.py -i PATH_TO_SRC_GPKG -o PATH_TO_DST_GPKG -s PATH_TO_SAMPLE_IMAGE -x SHIFT_PIXELS_X -y SHIFT_PIXELS_Y
   ```
   Shift using spatial units (SRS):
   ```
   python shift.py -i PATH_TO_SRC_GPKG -o PATH_TO_DST_GPKG -x SHIFT_UNITS_X -y SHIFT_UNITS_Y
   ```

ℹ️ Tip: For advanced settings, refer to the instructions in [delineation_config_guide.md](delineation_config_guide.md)


## License
This project is licensed under the AGPL-3.0 License.

## Acknowledgements
This code is based on [Ultralytics](https://github.com/ultralytics/ultralytics).

## Citation
If you find our work useful in your research, please consider citing it:
```
@inproceedings{lavreniuk2026delanyv2,
      title={Delineate Anything v2: A Global Foundation Model for Field Delineation}, 
      author={Mykola Lavreniuk and Nataliia Kussul and Andrii Shelestov and Yevhenii Salii and Volodymyr Kuzin and Charlotte Julia Li-Xing Wang and Zoltan Szantoi},
      year={2026},
      booktitle={European Conference on Computer Vision Workshops (ECCVW)},
}

@inproceedings{lavreniuk2025delineateanything,
      title={Delineate Anything: Resolution-Agnostic Field Boundary Delineation on Satellite Imagery}, 
      author={Mykola Lavreniuk and Nataliia Kussul and Andrii Shelestov and Bohdan Yailymov and Yevhenii Salii and Volodymyr Kuzin and Zoltan Szantoi},
      year={2025},
      booktitle={European Conference on Artificial Intelligence},
}

@article{lavreniuk2025delineateanythingflow,
      title={Delineate Anything Flow: Fast, Country-Level Field Boundary Detection from Any Source}, 
      author={Mykola Lavreniuk and Nataliia Kussul and Andrii Shelestov and Yevhenii Salii and Volodymyr Kuzin and Sergii Skakun and Zoltan Szantoi},
      year={2025},
      journal={https://arxiv.org/abs/2511.13417},
}
```
