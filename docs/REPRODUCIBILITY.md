# Reproducibility

The preserved scripts use 10 classes, four-trial pseudotrials, five-sample (20 ms) windows/steps at 250 Hz, five-fold CV, 10 repetitions, training-fold normalization, and no PCA for within-mode decoding. Directional cross-mode decoding retains 95% PCA variance and balances trials with participant-derived seeds. The classifier and NeuroRA calls are unchanged.

Within-mode decoding significance uses 1,000 sign-flip permutations, positive tail, cluster-forming p=.05, cluster α=.05, and seed 42. Latency uses 1,000 participant-level label permutations generated from seed 42; onset requires two consecutive post-onset bins above the pointwise 95th percentile null. Pairwise latency tests are two-sided with Bonferroni correction.

RSA uses correlation-distance 10×10 RDMs in five-sample windows. Within-mode positive 1D cluster tests use 1,000 permutations and seed 20260819. Paired mode differences and Figure 4 analyses use 1,000 permutations and seed 20260823; positive 2D threshold is t=1.6991 and two-sided threshold is |t|=2.0452 (df=29). sEMG-controlled cross-mode tests use 1,000 permutations and seed 20260824 with the same respective thresholds. Cluster α is .05. Correction is within each curve/map; no global correction across features or panels is applied.

Selected group curves may use PCHIP interpolation for display only. RSA inference uses the original 50 bins. The five-point decoding smoothing precedes the relevant curve-level inference. The archived within-mode temporal-generalization plot did not preserve its NeuroRA default permutation count/side setting; exact reproduction of that historical contour remains to be confirmed against the original NeuroRA environment.
