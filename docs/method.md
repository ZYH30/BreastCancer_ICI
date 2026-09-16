# Candidate-role-aware conditional benefit analysis

Let `A` denote ICI addition, `Y` the binary favorable pathological outcome, `B`
the candidate marker, and `G` the reference context. The clinical application
uses baseline CXCL9 and IFNG, respectively. Let

```text
tau(X) = E[Y(1) - Y(0) | X]
m(G)   = E[B | G]
g(G)   = E[tau(X) | G]
r      = B - m(G)
Psi    = E[r * (tau(X) - g(G))] / E[r^2].
```

This is signed residual treatment-benefit information. A zero value does not
exclude symmetric or higher-order effect modification. It is not a gene
intervention effect, and conditioning on IFNG does not remove every immune
program or establish biological independence.

## Adjustment interface

Native target-local discovery supplies a working pretreatment graph. Required
candidate/context measurements remain in the analysis without fabricating
marker-to-outcome edges. The role interface verifies backdoor admissibility
and treatment separation conditional on the propensity covariates. Importantly,
it covers the raw sources actually used by the outcome regression `Q`, not
only the named biomarker.

The candidate-leaf reduction operates on a pretreatment DAG with treatment
outgoing edges removed and formal leaves attached to required measurements.
`CandidateRoleLift` implements the associated joint-separation certificate and
records its conditions. Greedy reduction is not a claim of globally
minimum-cardinality adjustment. A certificate is conditional on the supplied
graph; it does not certify the correctness of data-learned arrows or the
absence of unobserved confounding.

## Estimation

The augmented score is

```text
phi = Q1 - Q0 + A*(Y-Q1)/e - (1-A)*(Y-Q0)/(1-e).
```

Outer cross-fitting separates training patients from evaluation patients.
Training-only inner scores fit `g`; their graph design is inherited from the
outer training set and is not represented as independent inner graph validation.
`m` and `g` use training-selected ridge/spline fits. The final common probability
bound is `[0.05, 0.95]` at both inner and outer stages; no clipping of `g` or
patient trimming is applied. Probability stabilization can introduce bias.

The estimated numerator and denominator are means of `r*(phi-g)` and `r^2`.
The implementation reports delta uncertainty, a zero-numerator score test, and
full Fieller-type confidence sets, including unbounded/disconnected cases.
Fieller inference and orthogonal scores are established tools, not inventions
of this software. Working small-sample inference is evaluated empirically;
asymptotic arguments do not guarantee finite-sample coverage.

## Comparisons

| Label | Interpretation |
| --- | --- |
| `M1` | All measured inputs in the propensity and outcome interfaces |
| `M3` | Candidate-preserving propensity closure covering actual Q inputs |
| `L2` | Graph and all nuisance components learned on training data |
| `L1` | True graph supplied; nuisance regressions still learned |
| `QR` | Restricted Q sources with the original propensity interface fixed |
| `complete_oracle` | True nuisance functions; diagnostic benchmark only |

The Q sensitivity gives both M1 and M3 the same restricted outcome inputs. Its
associated `g` is refitted to the changed inner score; `e`, `m`, patient/sample
sets, graph realization, and folds remain fixed. Oracle replacements isolate
error components and are not deployable clinical methods.

Clinical coefficients are scaled by each observed cohort's original empirical
gene-expression SD, not a common gene-intervention dose. The original 18-marker
family is retained for multiplicity control. An equal-weight two-study mean and
a conjunction test answer different questions; significance of the mean does
not establish separate confirmation in both studies.

I-SPY2 historical-arm assignment and NeoTRIP RNA/analysis selection remain
identification limitations. The NeoTRIP probability of one half is a declared
working randomization assumption, not proof of unbiased selection into the
measured subset.
