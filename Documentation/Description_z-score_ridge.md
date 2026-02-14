Think of your setup as a standard binary-choice panel model:

Outcome y = crisis/no crisis.
Regressor = Z-score (standardized, so “1 unit” = 1 standard deviation).
Plus country and year fixed effects (a dummy for almost every country and every year).
Plus class weights to give crises more influence (because crises are rare).
What went wrong in the non-ridge logit (the one that didn’t converge)
In basic econometrics terms, you hit the logit version of “perfect fit / collinearity problems”:

With tons of fixed-effect dummies, the model can often perfectly predict many observations (your output says ~91% can be perfectly predicted).
When that happens, maximum likelihood tries to push some coefficients toward very large magnitudes to make predicted probabilities basically 0 or 1.
Result: non-convergence, huge standard errors, p-values near 1, and coefficients that are not economically credible. You can’t trust magnitudes (and often not inference at all).
What ridge is doing (and why it’s ‘better’ here)
Ridge is just regularization/shrinkage:

It adds a penalty that discourages coefficients (especially the many fixed-effect dummies) from becoming huge.
That prevents the “blow up to infinity” behavior under quasi-separation.
So you get stable, finite estimates that you can actually interpret.
How to interpret the ridge result in simple terms
Your ridge estimate for Z-score is about -1.09, so:

A 1 standard deviation higher Z-score is associated with lower crisis risk, holding country and year effects constant.
In odds-ratio terms: exp(-1.09) ≈ 0.34, meaning the odds of crisis are about 66% lower (since 1 − 0.34 = 0.66).
Why the ridge coefficient is smaller in magnitude than the non-ridge one
The non-ridge model was partly “cheating” by pushing coefficients to extremes to match the data (because of separation). Ridge forces a more moderate fit, so the Z-score effect typically shrinks toward 0 (here from about -2.29 to -1.09).

Which fit metric to trust

The near-perfect training AUC (~0.99) is not reassuring with two-way fixed effects; it’s consistent with overfitting/perfect prediction.
The cross-validated performance (your CV AUC around ~0.91) is much more meaningful.
Net: ridge is better because it turns an ill-posed/non-convergent fixed-effects logit into a well-posed estimation problem, so the sign and magnitude of the Z-score effect are interpretable.