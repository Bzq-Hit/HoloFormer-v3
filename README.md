# Untrained Physics-Enhanced Fully Complex Transformer for Single-Frame Hologram Reconstruction (Optics Letters 2026)

Ziqi Bai, [Xianming Liu](https://homepage.hit.edu.cn/xmliu), [Cheng Guo](https://scholar.google.com.hk/citations?hl=zh-CN&user=D_jtz9sAAAAJ&view_op=list_works), [Junjun Jiang](https://homepage.hit.edu.cn/jiangjunjun?lang=zh), [Xiangyang Ji](https://www.au.tsinghua.edu.cn/info/1111/1524.htm)

---

Paper link:

- https://opg.optica.org/ol/abstract.cfm?uri=ol-51-7-1776
- DOI: 10.1364/OL.589365

accepted by Optics Letters (OL)

---

*Single-frame hologram reconstruction without training remains challenging for real-world samples due to the difficulty of modeling coupled amplitude and phase information. To address this issue, we develop a fully complex-domain attention mechanism that reformulates attention computation directly in the complex field using complex inner products, enabling more effective representation of amplitude–phase coupling. Based on this formulation, a general fully complex-domain transformer, termed HoloFormer v3, is constructed and instantiated within an untrained reconstruction framework. An auxiliary physics-based inverse diffraction module reformulates the highly nonlinear hologram-to-object complex amplitude inversion as a twin-image–suppressed optimization in the object plane. Experiments on real samples demonstrate improved reconstruction accuracy and faster convergence compared with existing untrained methods.*

![Image text](https://github.com/Bzq-Hit/HoloFormer-v3/blob/main/fig.png)

---

## Requirements

- Python 3
- PyTorch
- CUDA-enabled GPU with more than 16 GB of memory

---
## Usage

1. Prepare your hologram images in `.png` or `.jpg` format and place them in:
   
   ```bash
   ./data/YOUR_DATA
   ```
2. Prepare the imaging-system parameters as a `.json` file. An example is provided at:
   
   ```bash
   data/data_unlabel_cell/params.json
   ```
3. Modify `parse_task` in `utils/general.py` to load your own dataset, and set the reconstruction parameters in:
   
   ```bash
   configs/sample.yaml
   ```
4. Run reconstruction with:
   
   ```bash
   python main.py --task_config configs/sample.yaml
   ```

The reconstructed results will be saved in `./results`.

---

## Citation

If you find HoloFormer v3 useful in your research, please consider citing:

```
@article{bai2026untrained,
  title={Untrained physics-enhanced fully complex transformer for single-frame hologram reconstruction},
  author={Bai, Ziqi and Liu, Xianming and Guo, Cheng and Jiang, Junjun and Ji, Xiangyang},
  journal={Optics Letters},
  volume={51},
  number={7},
  pages={1776--1779},
  year={2026},
  publisher={Optica Publishing Group}
}
```

---

## Acknowledgements

This implementation builds upon the untrained framework provided by [yp000925/J-Net](https://github.com/yp000925/J-Net). We sincerely thank the authors for making their code publicly available.

---
