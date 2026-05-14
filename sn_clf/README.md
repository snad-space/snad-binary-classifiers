Related work: [Semenikhin et al. (2026)](https://doi.org/10.1016/j.ascom.2026.101126)

`bts_sample_png` contains all lightcurve images from BTS sample for SNe, which are in our ZTF DR23 dataset.
We visually inspected all of these curves, and then our sample was reduced to 674 SNe.

`notebooks/` includes the main code:

* `train_clf_BTS_sample.ipynb` -- notebook for training the SN vs non-SN classifier.
* `make_train_dataset.ipynb` -- code used to construct the training dataset.
* `pineforest_run.ipynb` -- experiments with running PineForest on different fields and feature sets.

Outputs from PineForest runs are stored in `pineforest_output/`. CSV tables contain two columns: oid (object identifier) and is_anomaly (0 for regular objects, 1 for anomalies, which in our case correspond to SNe).

Features extracted from light curves for PineForest experiments are available on [Zenodo](https://zenodo.org/records/17963500).

`models/SNclf_dr23.onnx` -- trained SN/non-SN classificator.
