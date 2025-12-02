# TimeSeriesCovMatcher - Time Series Covariate Matching (project in development)

TimeSeriesCovMatcher is a time-series–safe distribution matching library designed for scientific data analysis, especially neurophysiology and behavioral pipelines where temporal structure must not be destroyed.

It implements a workflow that preserves **quantile structure**, **temporal contiguity**, and **fixed-size block resampling**, all at once.

This method grew out of real-world problems in matching arousal distributions (via pupil size) before comparing neural metrics (like burst rates). It generalizes cleanly beyond neuroscience.

---

## Why this library?

Simple shuffling breaks temporal autocorrelation.  
Simple block resampling breaks state distributions.  
Simple quantile matching destroys transitions.

**TimeSeriesCovMatcher keeps all three intact**:

1. Per-sample quantile assignment  
2. Contiguous quantile-homogeneous segments (“qblocks”)  
3. Subdivision into fixed-length temporal tiles (“subblocks”)  
4. Quantile-wise matching by balanced subblock subsampling

This is the *only* pattern that:
- preserves long- and short-timescale structure,
- respects slow transitions (like arousal drift),
- ensures fair comparison across experimental groups.

---

## Core Concepts

### 1. Sample-level quantiles
Quantiles are computed per user-defined strata (e.g. batch × experiment type).  
This prevents mixing distributions across heterogeneous conditions.

### 2. qblocks = contiguous quantile segments
Within each experiment, samples with the same quantile form a qblock.  
Time gaps above a threshold start new qblocks.

This preserves natural transitions and avoids smearing signal across state changes.

### 3. Subblocks = fixed-size temporal tiles inside qblocks
Each qblock is subdivided into tiles of fixed length (e.g. 26 samples).  
Leftovers are dropped.

The subblock is the atomic matching unit.

### 4. Quantile-wise subblock matching
Within each matching stratum (e.g. batch × e_name × state):
- Compute quantile proportions for each experiment
- Determine a target = minimum proportion across experiments
- Keep all subblocks from the reference experiment
- Downsample other experiments proportionally

The resulting matched dataset preserves time structure while enforcing comparable covariate distributions.

---

## Usage Example

```python
from quantile_block_matcher import QuantileBlockMatcher

matcher = QuantileBlockMatcher(
    group_col="cond",
    covariate_col="area",
    time_col="ts",
    unit_cols=["m", "s", "e"],
    quantile_strata_cols=["batch", "e_name"],
    match_strata_cols=["batch", "e_name", "state"],
    n_quantiles=4,
    gap_threshold=0.05,
    block_len=26,
    random_state=0,
)

matched_df = matcher.fit_transform(df)
```

Here:

- df contains your time series data (e.g. pupil trace)
- cond is your group label (test/control)
- area is your covariate (pupil area)
- ts is timestamp
- m, s, e are experiment identifiers
- batch, e_name, state define matching granularity

The output keeps:

- quantile
- qblock_id
- subblock_id

for inspection and debugging.

## Scientific Rationale

This method was originally developed to:

- equalize arousal (pupil size) distributions across animals
- avoid over-representing specific behavioral states
- guarantee matched conditions before computing neural metrics
- preserve slow drift & transition phases
- avoid time-warping the signal

Most resampling approaches destroy essential time dependence.
QuantileBlockMatcher preserves the physics of the data.

## Arguments:

- group_col — group/condition column
- covariate_col — covariate to match
- time_col — timestamp
- unit_cols — experiment identifiers (e.g. mouse, session, experiment)
- quantile_strata_cols — strata for computing quantiles
- match_strata_cols — strata for subblock-level matching
- n_quantiles — number of quantile bins
- gap_threshold — max time gap allowed inside qblocks
- block_len — length of each subblock
- random_state — reproducible sampling

## Methods:

- fit(df)
- transform(df)
- fit_transform(df)

## License

MIT — free for academic and commercial use.
