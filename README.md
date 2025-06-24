# H-MagNet

**Spectral Synthesis of Magnetically Sensitive Fe Lines in the H Band Using Neural Networks**

H-MagNet is a family of neural‑network models that generate high‑resolution stellar spectra of Fe lines in the H band:

* **15 200.35 Å – 15 214.62 Å**
* **15 286.35 Å – 15 303.60 Å**
* **15 614.38 Å – 15 639.58 Å**

(all sampled every **~0.04 Å**).

Given five astrophysical parameters—effective temperature, surface gravity, global metallicity, magnetic field, and projected rotational velocity—the network outputs a synthetic flux vector with 1 328 points. Four model sizes let you trade accuracy for speed and memory.

The core library is light‑weight; the first time you instantiate a model, its weights are fetched from **Google Drive** via `gdown` and cached locally under `~/HMagNet_models/`.

Additionally, H-MagNet supports spectrum inversion, allowing users to estimate the underlying astrophysical parameters from observed or synthetic flux vectors using the neural network models in combination with optimization algorithms such as Particle Swarm Optimization (PSO).

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
| **tiny**   | ------     | ≈ 4 MB          | Edge devices & rapid scans     |
| **small**  | ------     | ≈ 26 MB         | Laptops / notebooks            |
| **medium** | ------     | ≈ 132 MB        | Workstations & small GPUs      |
| **large**  | \~45 M     | ≈ 516 MB        | Maximum fidelity (server GPUs) |

\*Sizes are approximate .h5 files downloaded on demand.

---

## Installation

> H-MagNet is currently in private beta. 


## Quick start 
### Synthetize — Single input
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

### Synthetize — Multiple inputs

```python
X = np.array([
    [6000, 4.0, 0.0, 0.4, 12],
    [4200, 3.0, –0.5, 3.0,  8],
    [3200, 3.8, 0.3, 4.0, 30],
])

synth = net.synthetize_spectra(X)
print(synth.shape)  # (3, 1328)
```
### Inversion - Estimate parameters from a spectrum
H-MagNet can also be used to invert spectra, estimating the five astrophysical parameters (Teff, logg, [M/H], Bfield, vsini) from an observed flux vector. It uses Particle Swarm Optimization (PSO) to minimize the error between the observed spectrum and the network's prediction.

```python
solution, inv_spectra, fitness = net.inversion(
    y_obs=spectrum,         # Input flux (shape: [1, 1328] or [1328])
    n_particles=1024,       # Number of particles (poblation size)
    iters=10,               # Optimization iterations
    verbose=1               # Show progress
)
```
This returns:

* **solution**: best-fit astrophysical parameters found by the optimizer.
* **inv_spectra**: synthetic spectrum generated from the inferred parameters.
* **fitness**: final value of the objective function.

### Custom objective function
You can provide your own objective function to compare the observed and predicted spectra. It must accept two arguments: y_obs and y_pred.

Example using mean absolute error (per wavelength point):

```python
from sklearn.metrics import mean_absolute_error
def obj(y_obs, y_pred):
    return mean_absolute_error(y_obs.T, y_pred.T, multioutput='raw_values')
```
Then use it like this:
```python
solution, inv_spectra, fitness = net.inversion(
    y_obs=spectrum,
    n_particles=1024,
    iters=10,
    objective_function=obj,
    verbose=1
)
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
wl   = net.get_wavelength()  # ndarray of 5503 wavelengths (Å)
seg  = net.get_segments()    # list of segment IDs ("0", "1", "2") for every region of spectra mentioned above
```

---

## API (`hmagnet`)

### `class HMagNet(model: str = "large")`

Main interface for spectral synthesis and inversion using neural network models.

---

### **Methods**

```python
synthetize_spectra(data: np.ndarray, batch_size: int = 32) -> np.ndarray
```

Generates synthetic spectra from stellar parameters.

* `data`: 1D or 2D NumPy array of shape `(5,)` or `(n_samples, 5)` in the order
  *(Teff, log g, \[M/H], Bfield, vsini)*.
* `batch_size`: number of inputs per batch (for efficiency on large inputs).

---

```python
get_wavelength() -> np.ndarray
```

Returns the wavelength grid (shape: `(1328,)`) used by the model.

---

```python
get_segments() -> list[int]
```

Returns the list of spectral segments covered by the model.
Each entry corresponds to one of the Fe line regions in the H band.

---

```python
inversion(
    y_obs: np.ndarray,
    n_particles: int,
    iters: int,
    objective_function: Callable = default_objective,
    W: float = 0.7,
    C1: float = 1.0,
    C2: float = 1.0,
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray, float]
```

Estimates stellar parameters from a given flux vector via Particle Swarm Optimization (PSO).

* `y_obs`: observed spectrum, shape `(1328,)` or `(1, 1328)`
* `n_particles`: number of particles in the PSO population
* `iters`: number of PSO iterations
* `objective_function`: custom error function of the form `f(y_obs, y_pred)`
  (default: mean squared error)
* `W`: inertia weight (controls momentum)
* `C1`: cognitive component (pull toward particle best)
* `C2`: social component (pull toward global best)
* `verbose`: `0` = silent, `1` = progress printed at each iteration

**Returns**:

* `solution`: inferred parameters (array of shape `(5,)`)
* `inv_spectra`: spectrum generated from the inferred parameters
* `fitness`: final value of the objective function

---

### Available model sizes

The `model` argument can be one of:

* `"tiny"` — fastest, lower accuracy
* `"small"` — good trade-off
* `"medium"` — higher accuracy, slower
* `"large"` *(default)* — best accuracy, highest memory and compute cost

Model weights are downloaded on first use and cached locally in `~/HMagNet_models/`.

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
