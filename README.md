# H-MagNet-example

**Spectral Synthesis of Magnetically Sensitive Fe Lines in the H Band Using Neural Networks**

H-MagNet is a family of neural‑network models that generate high‑resolution stellar spectra of Fe lines in the H band:

* **15 200.35 Å – 15 214.62 Å**
* **15 286.35 Å – 15 303.60 Å**
* **15 614.38 Å – 15 639.58 Å**

(all sampled every **~0.04 Å**).

Given five astrophysical parameters—effective temperature, surface gravity, global metallicity, magnetic field, and projected rotational velocity—the network outputs a synthetic flux vector with 1 328 points. Four model sizes let you trade accuracy for speed and memory.

The core library is light‑weight; the first time you instantiate a model, its weights are fetched from **Google Drive** via `gdown` and cached locally under `~/HMagNet_models/`.

---

## Astrophysical parameter space

| Parameter              | Min   | Max    |
| ---------------------- | ----- | ------ |
| **Teff** (K)           | 3 000 | 6 000  |
| **log g** (dex)        | 3.0   | 5.0    |
| **\[M/H]** (dex)       | –0.5  | +0.5   |
| **Bfield** (kG)        | 0.0   | 12.0   |
| **v sin i** (km s⁻¹)   | 0.0   | 35.0   |

---

## Model variants

| Variant    | Parameters | Download size\* | Best suited for                |
| ---------- | ---------- | --------------- | ------------------------------ |
| **tiny**   | ------     | ------          | Edge devices & rapid scans     |
| **small**  | ------     | ------          | Laptops / notebooks            |
| **medium** | ------     | ------          | Workstations & small GPUs      |
| **large**  | \~45 M     | ≈ 516 MB        | Maximum fidelity (server GPUs) |

\*Sizes are approximate .h5 files downloaded on demand.

The medium, small and tiny sizes are coming soon.

---

## Installation

> H-MagNet is currently in private beta. 


## Quick start — single input

```python
import numpy as np
from hmagnet import HMagNet

# Large‑size network (downloads weights on first run)
net = HMagNet("large")

# teff, logg, mh, bfield, vsini
x = np.array([5000, 4.25, 0.1, 5, 15])

spectrum = net.synthetize_spectra(x)
print(spectrum.shape)  # (1, 1328)
```

### Multiple inputs

```python
X = np.array([
    [6000, 4.0, 0.0, 0.4, 12],
    [4200, 3.0, –0.5, 3.0,  8],
    [3200, 3.8, 0.3, 4.0, 30],
])

synth = net.synthetize_spectra(X)
print(synth.shape)  # (3, 1328)
```

### Important
Be sure to select the corresponding device for run the model, whether GPU or CPU.
If using the CPU, you can select the number of jobs to use. This configuration must be done before instantiating HMagNet.

```python
from hmagnet import HMagNet, config
config(jobs=6)
net = HMagNet("large")
...
```

---

## Extra utilities

```python
wl   = net.get_wavelength()  # ndarray of 1328 wavelengths (Å)
seg  = net.get_segments()    # list of segment IDs ("0", "1", "2") for every region of spectra mentioned above
```

---

## API (`hmagnet`)

```python
class HMagNet(model: str = "large"):
    synthetize_spectra(data, batch_size: int = 32) -> np.ndarray
    get_wavelength() -> np.ndarray
    get_segments() -> list[str]
```

* **model** — one of `"tiny"`, `"small"`, `"medium"`, `"large"`
* **data** — 1‑D or 2‑D array with 5 columns in the order *(Teff, log g, \[M/H], Bfield, v sin i)*.
* **batch_size** — number of instances in each batch to be synthesized in the neural network.

---

## Troubleshooting

| Symptom                                          | Fix                                                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| *`ModuleNotFoundError: No module named 'gdown'`* | `pip install gdown`                                                                                                 |
| Slow download / blocked                          | Download the `.h5` manually from the Google Drive link in `_MODEL_TABLE` and place it under `~/HMagNet_models/`. |
| `ValueError: Model 'xyz' not exist.`             | Use one of the four valid model names.                                                                              |

---

## Contributing

Bug reports and feature requests are welcome on the *Issues* page.

---

## License

MIT © 2025 Joan Raygoza
