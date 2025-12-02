"""
Quantile-block matching for time series covariates.

    1. Assign each sample to a quantile of some covariate (e.g. pupil area),
       computed within a given strata (e.g. batch × experiment type).

    2. Within each quantile, partition the time series into contiguous
       "qblocks" – segments of consecutive samples (in time) that belong
       to the same quantile and are not separated by large time gaps.

    3. Split each qblock into fixed-length "subblocks" (e.g. 26 samples).
       Leftover samples at the tail of a qblock that don't fit into a full
       subblock are dropped.

    4. For each matching stratum (e.g. batch × experiment type × state),
       and for each quantile, subsample subblocks across groups/experiments
       so that:
           - the *within-experiment* quantile proportions are matched
             to the minimum across experiments in that stratum.

This is essentially a time-series-safe distribution matcher that operates
on quantile-homogeneous subblocks instead of raw samples or arbitrary blocks.

Typical usage (similar in spirit to your PupilData workflow)
------------------------------------------------------------
Suppose df has columns:
    - 'cond'     : group label (e.g. 'test' vs 'control')
    - 'area'     : covariate (e.g. pupil area)
    - 'ts'       : time stamp
    - 'batch'    : batch id
    - 'e_name'   : experiment type
    - 'state'    : behavioral state (e.g. 'sitting', 'running')
    - 'm','s','e': experiment identity (mouse, series, experiment)

You want to:
    - estimate quantiles on area per (batch, e_name)
    - then, *within each* (batch, e_name, state) stratum,
      match test/control pupil distributions by subsampling subblocks.

Example
-------
>>> matcher = QuantileBlockMatcher(
...     group_col="cond",
...     covariate_col="area",
...     time_col="ts",
...     unit_cols=["m", "s", "e"],
...     quantile_strata_cols=["batch", "e_name"],
...     match_strata_cols=["batch", "e_name", "state"],
...     n_quantiles=4,
...     gap_threshold=0.05,
...     block_len=26,
...     random_state=0,
... )
>>> matched_df = matcher.fit_transform(df)

Then `matched_df` contains only those samples belonging to selected subblocks,
and test/control have matched quantile distributions (per match stratum),
without breaking temporal structure at scales ≤ block_len.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


StratumKey = Tuple[Hashable, ...]


@dataclass
class QuantileBlockMatcher:
    """
    Quantile-based time-series matching using contiguous blocks.

    Parameters
    ----------
    group_col : str
        Column indicating the high-level group (e.g. 'cond': test vs control).
    covariate_col : str
        Column containing the covariate whose distribution you want to match
        across groups (e.g. pupil area).
    time_col : str
        Column with time stamps (used to detect contiguity and time gaps).
    unit_cols : Sequence[str]
        Columns that uniquely identify an "experiment" or unit for matching
        (e.g. ['m', 's', 'e'] for mouse/series/experiment).
    quantile_strata_cols : Sequence[str]
        Columns defining strata over which quantiles are computed.
        Example: ['batch', 'e_name'] – quantiles can differ per batch & experiment type.
    match_strata_cols : Sequence[str]
        Columns defining strata within which *matching* is performed.
        Example: ['batch', 'e_name', 'state'].
        Within each such stratum, all groups/experiments are matched.
    n_quantiles : int, default 4
        Number of quantile bins.
    gap_threshold : float, default 0.0
        Time gap threshold (in units of `time_col`) used to split qblocks.
        If consecutive samples are separated by more than this, a new qblock
        is started (within the same quantile).
    block_len : int, default 26
        Length (in samples) of subblocks into which qblocks are tiled.
        Samples at the tail of a qblock that don't fit into a full subblock
        are dropped.
    random_state : int, optional
        Seed for reproducible subblock sampling.

    Notes
    -----
    The workflow is:

        df
          → assign_quantiles()    # per quantile_strata_cols
          → annotate_blocks()     # qblocks + subblocks within each quantile
          → match_distribution()  # subsample subblocks per match_stratum

    The `fit` / `transform` / `fit_transform` interface is provided but this
    is not a typical estimator in the scikit-learn sense; all state is
    derived from the given dataframe.
    """

    group_col: str
    covariate_col: str
    time_col: str
    unit_cols: Sequence[str]
    quantile_strata_cols: Sequence[str]
    match_strata_cols: Sequence[str]
    n_quantiles: int = 4
    gap_threshold: float = 0.0
    block_len: int = 26
    random_state: Optional[int] = None

    # internal RNG (for reproducibility)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "QuantileBlockMatcher":
        """
        For compatibility; currently stateless, so this just validates the input.
        """
        self._validate_columns(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full matching pipeline to df and return the subsampled result.
        """
        self._validate_columns(df)

        df = df.copy()
        df = self._assign_quantiles(df)
        df = self._annotate_blocks(df)
        matched = self._match_distribution(df)
        return matched

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience wrapper around fit().transform().
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internal: column checks
    # ------------------------------------------------------------------
    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = (
            [self.group_col, self.covariate_col, self.time_col]
            + list(self.unit_cols)
            + list(self.quantile_strata_cols)
            + list(self.match_strata_cols)
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {missing}")

    # ------------------------------------------------------------------
    # Step 1: assign quantiles per quantile_stratum
    # ------------------------------------------------------------------
    def _assign_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a quantile index to each sample, computed within
        quantile_strata_cols.
        """
        df = df.copy()

        def _assign_in_group(group: pd.DataFrame) -> pd.DataFrame:
            cov = group[self.covariate_col]
            if cov.notna().sum() == 0:
                group["quantile"] = np.nan
                return group

            # Quantile edges from 0..1
            qs = np.linspace(0.0, 1.0, self.n_quantiles + 1)
            edges = cov.quantile(qs).values
            edges = self._make_strictly_increasing(edges)

            labels = list(range(self.n_quantiles))  # 0..n-1
            group["quantile"] = pd.cut(
                cov,
                bins=edges,
                labels=labels,
                include_lowest=True,
                duplicates="drop",
            )
            return group

        df = df.groupby(list(self.quantile_strata_cols), group_keys=False).apply(_assign_in_group)
        return df

    # ------------------------------------------------------------------
    # Step 2: qblocks + fixed-length subblocks
    # ------------------------------------------------------------------
    def _annotate_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each (match_strata, group, unit, quantile) combination:

            1. Sort by time.
            2. Split into contiguous qblocks based on time gaps > gap_threshold.
            3. Within each qblock, tile into subblocks of length block_len.
               Samples that don't fit into a full subblock are dropped.

        Adds columns:
            - 'qblock_id'   : contiguous quantile block index
            - 'subblock_id' : subblock index within each experiment/quantile
        """
        df = df.copy()

        group_cols = list(self.match_strata_cols) + [self.group_col] + list(self.unit_cols)

        def _process_group(group: pd.DataFrame) -> pd.DataFrame:
            # Work per quantile
            # Ignore rows with NaN quantile
            if group["quantile"].isna().all():
                return group.iloc[0:0]  # empty

            out_chunks: List[pd.DataFrame] = []
            for q in sorted(group["quantile"].dropna().unique()):
                qslice = group[group["quantile"] == q].copy()
                if qslice.empty:
                    continue

                # 2a. find qblocks by time gaps within this quantile
                qslice = self._find_qblocks(qslice)

                # 2b. within each qblock, create subblocks of fixed length
                qslice = self._create_subblocks(qslice)
                if qslice.empty:
                    continue

                out_chunks.append(qslice)

            if not out_chunks:
                return group.iloc[0:0]  # empty group
            return pd.concat(out_chunks, axis=0)

        df = df.groupby(group_cols, group_keys=False).apply(_process_group)

        return df

    def _find_qblocks(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Given a group with constant quantile, split into contiguous qblocks
        using time gaps > gap_threshold.
        """
        group = group.sort_values(self.time_col).copy()
        t = group[self.time_col].to_numpy()
        if len(t) == 0:
            group["qblock_id"] = np.array([], dtype=int)
            return group

        if len(t) == 1 or self.gap_threshold <= 0:
            # Single block if no gap threshold or only one sample
            group["qblock_id"] = 0
            return group

        dt = np.diff(t)
        # indices where a new block starts
        split_points = np.where(dt > self.gap_threshold)[0] + 1

        block_ids = np.zeros(len(t), dtype=int)
        for i, sp in enumerate(split_points):
            block_ids[sp:] += 1

        group["qblock_id"] = block_ids
        return group

    def _create_subblocks(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Within each qblock, create fixed-length subblocks.

        For each (qblock_id), we:
            - take samples in time order
            - keep only the first n_full = floor(len / block_len) * block_len samples
            - assign subblock_ids 0..(n_blocks-1), each repeated block_len times

        Returns a DataFrame containing only samples that belong to a full subblock.
        """
        group = group.copy()
        out_chunks: List[pd.DataFrame] = []

        for qb in sorted(group["qblock_id"].unique()):
            qb_slice = group[group["qblock_id"] == qb].sort_values(self.time_col).copy()
            n = len(qb_slice)
            if n < self.block_len:
                continue  # too short to form one subblock

            n_blocks = n // self.block_len
            n_full = n_blocks * self.block_len
            qb_slice = qb_slice.iloc[:n_full].copy()

            subblock_ids = np.repeat(np.arange(n_blocks), self.block_len)
            qb_slice["subblock_id"] = subblock_ids
            out_chunks.append(qb_slice)

        if not out_chunks:
            return group.iloc[0:0]

        return pd.concat(out_chunks, axis=0)

    # ------------------------------------------------------------------
    # Step 3: subsample subblocks to match quantile distributions
    # ------------------------------------------------------------------
    def _match_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each match_stratum (e.g. batch × e_name × state):

            1. Compute, for each experiment (group + unit), the proportion of
               samples in each quantile.

            2. For each quantile q, find the minimum proportion across all
               experiments: this is the target proportion for that q.

            3. For each experiment and quantile:
                 - choose a "reference experiment" with the minimum proportion,
                   keep all its subblocks for that quantile.
                 - for all other experiments, randomly subsample subblocks
                   to reach (target_prop / prop_experiment) fraction of blocks.

        """
        df = df.copy()

        # we assume subblock_id is already present; drop any samples that aren't in full subblocks
        if "subblock_id" not in df.columns:
            raise RuntimeError("subblock_id not found. Did you run _annotate_blocks first?")
        df = df.dropna(subset=["subblock_id"])

        # match within each match_stratum (e.g. batch, e_name, state)
        def _subsample_stratum(group: pd.DataFrame) -> pd.DataFrame:
            if group.empty:
                return group

            # 1) quantile counts per experiment
            exp_cols = [self.group_col] + list(self.unit_cols)
            q_cols = exp_cols + ["quantile"]

            counts = (
                group.groupby(q_cols)
                .size()
                .rename("count")
                .reset_index()
            )
            totals = (
                group.groupby(exp_cols)
                .size()
                .rename("total")
                .reset_index()
            )

            props = counts.merge(totals, on=exp_cols)
            props["prop"] = props["count"] / props["total"]

            if props.empty:
                return group.iloc[0:0]

            # 2) target proportion per quantile = minimum across experiments
            target_props = props.groupby("quantile")["prop"].min()

            # Pre-index for speed
            group_out_chunks: List[pd.DataFrame] = []

            # 3) loop over quantiles
            for q in sorted(group["quantile"].dropna().unique()):
                q_slice = group[group["quantile"] == q].copy()
                if q_slice.empty:
                    continue

                qprops = props[props["quantile"] == q].copy()
                if qprops.empty:
                    continue

                target_prop = target_props.loc[q]

                # If target_prop is zero (everyone has zero?), skip quantile
                if target_prop <= 0:
                    continue

                # Compute rel = target_prop / prop for each experiment
                qprops["rel"] = target_prop / qprops["prop"]
                # Identify reference experiment (min prop)
                ref_idx = qprops["prop"].idxmin()
                ref_row = qprops.loc[ref_idx]
                ref_experiment = tuple(ref_row[c] for c in exp_cols)

                # Merge rel into q_slice
                rel_cols = exp_cols + ["rel"]
                q_slice = q_slice.merge(
                    qprops[rel_cols],
                    on=exp_cols,
                    how="inner",
                )

                # Full slice: all subblocks of the reference experiment
                ref_mask = np.logical_and.reduce(
                    [q_slice[c] == v for c, v in zip(exp_cols, ref_experiment)]
                )
                full_slice = q_slice[ref_mask].copy()

                # Complement: other experiments
                complement = q_slice[~ref_mask].copy()

                # Subsample complement by experiment (unit_cols) using rel
                if not complement.empty:
                    subsampled = (
                        complement.groupby(self.unit_cols, group_keys=False)
                        .apply(self._subsample_in_experiment)
                    )
                    group_out_chunks.append(subsampled)

                if not full_slice.empty:
                    group_out_chunks.append(full_slice)

            if not group_out_chunks:
                return group.iloc[0:0]
            return pd.concat(group_out_chunks, axis=0)

        matched = df.groupby(list(self.match_strata_cols), group_keys=False).apply(_subsample_stratum)
        # Drop any helper columns if you don't want them in final result
        # (keeping quantile/subblock_id is often useful for diagnostics, so we leave them)
        return matched

    def _subsample_in_experiment(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Subsample subblocks within a single experiment for a given quantile.

        The group here corresponds to:
            one experiment (unit_cols) × one quantile × one match_stratum (outer),
        but possibly multiple groups (cond) if you use unit_cols without cond.

        We:
            - compute unique subblocks in this group
            - total_blocks = n_unique_subblocks
            - target_blocks = floor(total_blocks * rel)
            - randomly pick that many subblocks
        """
        if group.empty:
            return group

        rel_vals = group["rel"].unique()
        if len(rel_vals) != 1:
            raise ValueError("Expected a single unique 'rel' value per experiment×quantile group.")
        proportion = rel_vals[0]

        unique_blocks = group["subblock_id"].unique()
        total_blocks = len(unique_blocks)
        if total_blocks == 0 or proportion <= 0:
            return group.iloc[0:0]

        target_blocks = int(np.floor(total_blocks * proportion))
        if target_blocks == 0:
            return group.iloc[0:0]

        chosen_blocks = self._rng.choice(unique_blocks, size=target_blocks, replace=False)
        block_mask = group["subblock_id"].isin(chosen_blocks)
        return group[block_mask]

    # ------------------------------------------------------------------
    # Small helper
    # ------------------------------------------------------------------
    @staticmethod
    def _make_strictly_increasing(bins: np.ndarray) -> np.ndarray:
        """
        Ensure quantile bin edges are strictly increasing (numerical safety).
        """
        bins = bins.astype(float)
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + 1e-9
        return bins

