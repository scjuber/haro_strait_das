# Haro Strait DAS

Whale call analysis of DAS recordings from the Haro Strait subsea cable.

- `haro-strait-DAS-analysis.ipynb`: main analysis (loop stacking, PSD/TX figures, spectrograms)
- `combined-DAS-event-analysis.ipynb`: SNR-enhancement comparison across events
- `manuscript_das_figures.ipynb`: manuscript spectrogram figures
- `das_functions.py`, `noise_env_sub.py`: shared functions

Raw data (`decimator_*.h5`) are not included; set `fdir`/`flist` to your data location.

```bash
pip install -r requirements.txt
```
