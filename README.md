# CLASS with Multiple Reionization Schemes and High-z Reionization Extension

This is a modified version of the **CLASS (Cosmic Linear Anisotropy Solving System)** code ([class_public-3.2.3] (https://github.com/lesgourg/class_public/tree/3.2.3)). It incorporates **multiple reionization schemes**, offering more flexible and data-driven approaches to studying the epoch of reionization. These schemes include:

* **Original reionization schemes in CLASS**: reio_camb, reio_bins_tanh, reio_many_tanh, and others (for details, see [class_public/explanatory.ini] (https://github.com/lesgourg/class_public/blob/master/explanatory.ini)).
* **A new reionization scheme called reio_gpr_tanh** (Gaussian Process Regression with a hyperbolic tangent function).

This modified code also introduces two new derived parameters:
* **tau_lowz and tau_highz**: These parameters extend reionization reconstruction and constraint to before recombination, splitting the total optical depth into contributions from both high and low redshift ranges.

## Key Modifications

This modified code introduces enhanced interpolation methods for reionization:
* **class_onen**: This method uses equally spaced bins for interpolation, primarily suited for the reconstruction of tau_lowz.
* **class_gp**: This method implements an adaptive bin interpolation approach, enabling the comprehensive calculation of tau_lowz, tau_highz, and tau_total.

For more detailed information, please refer to our upcoming paper: **Cheng et al. (2025)**.

The main modifications to the original CLASS code are in input.c, thermodynamics.c, and thermodynamics.h. These changes are clearly marked with "Hanyu" in the comments.

## Citation

This code is freely available for use. If you use this code in your research, please cite both:
* **Cheng et al. (2025)** (to be released soon)
* The original CLASS release paper: [**Blas et al. (2011)**] (https://arxiv.org/abs/1104.2933)

## Future Releases

The results folder, which will include related test Python files, .yaml files, and chains, will be released soon.
