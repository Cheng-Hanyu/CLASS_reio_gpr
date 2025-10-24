# CLASS with Multiple Reionization Schemes and High-z Reionization Extension

This is a modified version of the [**CLASS**](https://github.com/lesgourg/class_public#) code (version class_public-3.2.3). It incorporates **multiple reionization schemes**, offering more flexible and data-driven approaches to studying the epoch of reionization. These schemes include:

* **Original reionization schemes in CLASS**: reio_camb, reio_bins_tanh, reio_many_tanh, and others (for details, see [class_public/explanatory.ini](https://github.com/lesgourg/class_public/blob/master/explanatory.ini)).
* **A new reionization scheme called reio_gpr_tanh** (Gaussian Process Regression with a hyperbolic tangent function).

This modified code also introduces two new derived parameters:
* **tau_lowz and tau_highz**: These parameters extend reionization reconstruction and constraint to before recombination, splitting the total optical depth into contributions from both high and low redshift ranges.

## Key Modifications
There are two different modified version. 
* **class_onen**: This version uses equally spaced bins for interpolation, primarily suited for the reconstruction of tau_lowz.
* **class_gp**: This version implements an adaptive bin interpolation approach, enabling the comprehensive calculation of tau_lowz, tau_highz, and tau_total.

For more detailed information, please refer to our paper: **Cheng et al. (2025)**: [arXiv:2506.19096](https://arxiv.org/abs/2506.19096).

The main modifications to the original CLASS code are in input.c, thermodynamics.c, and thermodynamics.h. These changes are clearly marked with "Hanyu" in the comments.

## Citation

This code is freely available for use. If you use this code in your research, please cite both:
* **Cheng et al. (2025)**: [arXiv:2506.19096](https://arxiv.org/abs/2506.19096)
* The original CLASS release paper: [**Blas et al. (2011)**](https://arxiv.org/abs/1104.2933)

## Results and Data

The following folders contain results and analysis code from **Cheng et al. (2025)**:

* **gpr_general**: Includes four converged MCMC chains (R < 0.1), the corresponding `.yaml` configuration file, and a shell script to reproduce the MCMC analysis.

* **gpr_plot**: Python scripts for reconstructing and visualizing the reionization history (X_e as a function of redshift z from z = 0 to z = 800) from the MCMC chains. Reproduces FIG 1 from the paper.

* **gpr_tau_zc**: Python scripts for computing posterior distributions of tau_lowz and tau_highz for different redshift cutoff values (z_c), and the total optical depth tau_total. Generates 1σ and 2σ posterior constraints corresponding to TABLE 3, FIG 2, and FIG S3 in the paper.
