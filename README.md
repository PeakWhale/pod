<img src="pod.svg" alt="PeakWhale Pod logo" width="112" height="112">

# PeakWhale Pod

### Regional case clustering

A historical notebook that groups Indian states and union territories by reported COVID-19 cases and deaths using K-means clustering.

Originally created in 2020 as `adrkv/covid19positivecases`, this project is now part of [PeakWhale](https://github.com/PeakWhale). Its original notebook, spreadsheet outputs, and commit history are preserved.

## What it does

- Explores reported case and death counts for Indian states and union territories.
- Plots the elbow curve and groups observations into three K-means clusters.
- Visualizes the clusters and their centroids.

The original README dates the analysis to **April 9, 2020**. This is an exploratory historical analysis, not a current public-health model. It groups regional counts; it does not use geographic coordinates or spatial-distance methods.

## Files

| File | Purpose |
| --- | --- |
| `coronavirus_india_lockdown.ipynb` | Original analysis notebook and saved outputs |
| `case1.xlsx`, `case2.xlsx`, `case3.xlsx` | Original cluster-output spreadsheets |

## Viewing and running

You can view the notebook and its saved outputs directly on GitHub. The notebook uses Python with pandas, scikit-learn, matplotlib, seaborn, and Jupyter.

**The original input file `corona_india_state.csv` is not included.** A full rerun requires restoring that historical source data and checking its provenance. Package versions were not recorded, so the environment has not been verified against current releases. The notebook and data outputs are retained as originally published.

## License

No explicit license was specified in the original repository. Moving this project into PeakWhale does not add a reuse license for its code or data.

## Author

[Addy Kaveti](https://adrkv.github.io/) · [PeakWhale](https://github.com/PeakWhale)
