# AM/FM Local Parameter Estimation in Python

![Python](https://img.shields.io/badge/python-3.9%2B-blue)  

This repository contains Python implementations for **local AM/FM parameter estimation** of audio signals, inspired by:

> Dominique Fourer, François Auger, Geoffroy Peeters,  
> "Local AM/FM parameters estimation: application to sinusoidal modeling and blind audio source separation,"  
> IEEE Signal Processing Letters, Vol. 25, Issue 10, pp. 1600-1604, Oct. 2018.  
> DOI: [10.1109/LSP.2018.2867799](https://doi.org/10.1109/LSP.2018.2867799)

The code allows generating AM/FM signals, computing STFTs, and estimating sinusoidal parameters including amplitude modulation (AM) and frequency modulation (FM).

---

## Repository Contents

| File | Description |
|------|-------------|
| `Example.py` | Decomposition of an **entire audio file** into AM/FM parameters. |
| `Example2.py` | Decomposition of a **long sinusoidal signal** using the AM/FM model. |
| `Example3.py` | Estimation of **local AM/FM parameters on a single frame**. |

---

## Installation

Requires Python 3.9+ and the following packages:

```bash
pip install numpy scipy matplotlib


## License

This repository is licensed under a **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
You may use, modify, and share this code for **non-commercial purposes** only.  

For details, see [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
