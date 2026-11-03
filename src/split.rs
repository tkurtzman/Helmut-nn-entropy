//! The split estimator: soft coordinates non-parametrically, hard ones with a
//! Gaussian.
//!
//! Multi-basin torsions are poorly described by a Gaussian, while bonds,
//! angles and phase angles are close to harmonic, and a Gaussian captures
//! their correlations to every order. The split estimator divides the
//! coordinates into a soft set S and a hard set H and writes
//!
//! S = S_soft + S_hard + coupling,
//!
//! - S_soft: the nearest-neighbor MIE, the mutual information spanning tree
//!   (MIST) or a joint nearest-neighbor estimate over the soft coordinates;
//! - S_hard: the quasi-harmonic (QH) or Boltzmann quasi-harmonic (BQH)
//!   entropy of the hard coordinates over all frames, or QH truncated at
//!   second order in the mutual information expansion;
//! - the coupling term, at most 0, for the dependence between the two sets:
//!   either the Gaussian mutual information between them, or the change in
//!   the hard entropy when it is computed within each conformational state of
//!   the soft torsions and averaged over the states.
//!
//! The soft set is given explicitly, or detected from the data with
//! [`detect_basins`]. For coordinates read from a topology,
//! [`InternalCoordinates::soft_torsions`](crate::bat_library::InternalCoordinates::soft_torsions)
//! gives the topological default: the primary torsion about each rotatable
//! bond.

use std::collections::{BTreeMap, HashMap};

use crate::basins::{BasinOptions, Basins, detect_basins};
use crate::quasi_harmonic::{
    MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions, Representation,
    coordinate_quasi_harmonic_terms, correlation_log_det, correlation_log_diagonal,
    quasi_harmonic_terms, truncated_quasi_harmonic_terms,
};
use crate::{
    CoordinateMetric, EntropyOptions, calculate_entropy, estimate_coordinate_entropy,
    estimate_coordinate_mist_entropy, estimate_coordinate_mutual_information,
    estimate_coordinate_network_entropy, estimate_joint_entropy, estimate_mist_entropy,
    estimate_network_entropy, resolve_input,
};
use rayon::prelude::*;

/// Above this many soft coordinates the front ends warn that the split
/// estimator is outside its validated range. It was accurate to about 1 nat
/// with up to 15 soft torsions (hexadecane), but on folded peptides with 50
/// to 99 the default was off by tens of nats or unstable across strides, and
/// MIST and BQH, though stable, were off by up to 5-10% of the entropy change.
pub const LARGE_SOFT_SET: usize = 20;

/// The front ends' warning for a soft set larger than [`LARGE_SOFT_SET`].
pub fn large_soft_set_warning(soft: usize) -> String {
    format!(
        "{soft} soft coordinates, more than the {LARGE_SOFT_SET} the split estimator was validated for; for folded or compact molecules, compare with MIST and BQH, and check the result on every 2nd and 4th frame"
    )
}

/// Which coordinates are soft.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoftSelection<'a> {
    /// Exactly these coordinates, by index.
    Indices(&'a [usize]),
    /// The periodic coordinates among `candidates` (all periodic coordinates
    /// when `None`) that [`Basins::is_soft`] calls soft: more than one basin,
    /// or a free rotor.
    Detect {
        /// Coordinates to consider; every one must be periodic.
        candidates: Option<&'a [usize]>,
    },
}

/// How the soft entropy is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftEstimator {
    /// The nearest-neighbor mutual information expansion at `order`, 1 to 4.
    NearestNeighborMie {
        /// Expansion order.
        order: usize,
    },
    /// One joint nearest-neighbor estimate over every soft coordinate, up to
    /// [`MAX_JOINT_DIMENSION`](crate::MAX_JOINT_DIMENSION) of them.
    JointNearestNeighbor,
    /// The mutual information spanning tree; see
    /// [`estimate_mist_entropy`].
    MutualInformationSpanningTree,
    /// A Bayesian network with up to `parents` parents per coordinate; see
    /// [`estimate_network_entropy`].
    BayesianNetwork {
        /// Parents per coordinate, 1 (MIST) to `MAX_JOINT_DIMENSION - 1`.
        parents: usize,
    },
}

impl Default for SoftEstimator {
    /// The MIE at order 2.
    fn default() -> Self {
        Self::NearestNeighborMie { order: 2 }
    }
}

/// How the hard entropy is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HardEstimator {
    /// A Gaussian fit to the hard coordinates (QH).
    #[default]
    QuasiHarmonic,
    /// Each hard coordinate's own entropy from its distribution, plus the
    /// Gaussian correlation term (BQH).
    BoltzmannQuasiHarmonic,
    /// QH truncated at first order in the mutual information expansion: the
    /// Gaussian entropy of each hard coordinate, with no correlation term, as
    /// if the hard coordinates were independent. The coupling to the soft
    /// coordinates is unchanged.
    QuasiHarmonicMie1,
    /// QH truncated at second order in the mutual information expansion: the
    /// Gaussian entropy of each hard coordinate, minus the Gaussian mutual
    /// information of each pair, 1/2 ln(1 - C_ij^2), in place of the full
    /// correlation term 1/2 ln|C|. For Gaussian coordinates the full term is
    /// exact; the pairwise one misses the dependence shared among three or more
    /// coordinates, and can be too high or too low.
    QuasiHarmonicMie2,
}

/// How soft torsions enter the Gaussian coupling term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftFeatures {
    /// As e^(i phi): one complex entry each.
    Complex,
    /// As cos(phi) and sin(phi): two real entries each. The same as
    /// `Harmonics(1)`.
    CosSin,
    /// As cos(k phi) and sin(k phi) for k = 1 to the given number of
    /// harmonics: two real entries per harmonic. Higher harmonics let a hard
    /// coordinate depend on which of three wells a torsion is in, not only on
    /// cos(phi) and sin(phi).
    Harmonics(usize),
}

impl Default for SoftFeatures {
    /// Three harmonics.
    fn default() -> Self {
        Self::Harmonics(DEFAULT_HARMONICS)
    }
}

/// Harmonics of the default [`SoftFeatures`].
pub const DEFAULT_HARMONICS: usize = 3;

/// Which conformational states the per-state coupling conditions on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateDefinition<'a> {
    /// The joint basin of the state-defining torsions: those in `defining`
    /// (the soft coordinates when `None`) not in `exclude`, keeping only those
    /// with at least two basins. Exclude symmetric rotors such as methyl
    /// groups, whose basins are equivalent.
    Detected {
        /// Candidate state-defining coordinates.
        defining: Option<&'a [usize]>,
        /// Coordinates that never define states.
        exclude: &'a [usize],
    },
    /// A state label for every frame, given by the caller.
    Labels(&'a [u32]),
}

/// Settings for the per-state coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateOptions<'a> {
    /// Which states to condition on.
    pub definition: StateDefinition<'a>,
    /// States with fewer frames than this, or than the number of hard
    /// coordinates plus 2, are dropped and the remaining populations
    /// renormalized.
    pub min_frames: usize,
    /// Fail when more states than this have at least `min_frames` frames.
    pub max_states: usize,
}

impl Default for StateOptions<'_> {
    /// Detected states from all soft coordinates, at least 300 frames per
    /// state, at most 256 states.
    fn default() -> Self {
        Self {
            definition: StateDefinition::Detected {
                defining: None,
                exclude: &[],
            },
            min_frames: 300,
            max_states: 256,
        }
    }
}

/// How the dependence between the soft and hard sets is accounted for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Coupling<'a> {
    /// Not at all: the soft and hard sets are treated as independent.
    None,
    /// The hard entropy within each conformational state, averaged over the
    /// states, in place of the hard entropy over all frames.
    PerState(StateOptions<'a>),
    /// The Gaussian mutual information between the two sets.
    Gaussian {
        /// How soft torsions enter the correlation matrix.
        features: SoftFeatures,
    },
    /// The nearest-neighbor pairwise mutual information between the two sets:
    /// the hard-soft cross terms of the second-order expansion, estimated the
    /// same way as the soft set's own pair terms and subtracted like them.
    ///
    /// This is the term that [`Coupling::None`] drops. It costs one
    /// two-dimensional nearest-neighbor search per hard-soft pair, so the work
    /// grows as the product of the two set sizes.
    NearestNeighborMie,
}

/// Settings for [`estimate_split_entropy`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitOptions<'a> {
    /// Which coordinates are soft.
    pub soft: SoftSelection<'a>,
    /// How the soft entropy is estimated.
    pub soft_estimator: SoftEstimator,
    /// How the hard entropy is estimated.
    pub hard_estimator: HardEstimator,
    /// Quasi-harmonic settings for the hard set: BQH marginals, and how hard
    /// periodic coordinates (phase angles) enter the correlation matrix.
    pub hard: QuasiHarmonicOptions,
    /// How the dependence between the sets is accounted for.
    pub coupling: Coupling<'a>,
    /// Settings for basin detection, used by [`SoftSelection::Detect`] and
    /// [`StateDefinition::Detected`].
    pub basins: BasinOptions,
}

impl<'a> SplitOptions<'a> {
    /// The default settings for the soft set `soft`: NN MIE-2 for the soft
    /// coordinates; QH for the hard ones, with hard periodic coordinates
    /// centered on their circular mean; and Gaussian coupling with soft
    /// torsions as cos(k phi) and sin(k phi), k = 1 to 3. These scored best
    /// overall against MBAR references for small molecules, including eight
    /// held out from the choice; see the validation page of the user guide.
    pub fn new(soft: SoftSelection<'a>) -> Self {
        Self {
            soft,
            soft_estimator: SoftEstimator::default(),
            hard_estimator: HardEstimator::default(),
            hard: QuasiHarmonicOptions {
                periodic: PeriodicCorrelation::Centered,
                ..Default::default()
            },
            coupling: Coupling::Gaussian {
                features: SoftFeatures::default(),
            },
            basins: BasinOptions::default(),
        }
    }
}

/// One conformational state of the per-state coupling.
#[derive(Debug, Clone, PartialEq)]
pub struct StateEntropy {
    /// Basin of each state-defining coordinate, or the caller's label.
    pub basins: Vec<u32>,
    /// Frames in the state.
    pub frames: usize,
    /// Fraction of all frames in the state.
    pub population: f64,
    /// Hard entropy within the state; `None` when the state was dropped.
    pub hard_entropy: Option<f64>,
}

/// The states of the per-state coupling.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSummary {
    /// The coordinates whose basins define the states; empty for labels given
    /// by the caller.
    pub defining: Vec<usize>,
    /// Every observed state, in order of its basins.
    pub states: Vec<StateEntropy>,
    /// Frames in dropped states.
    pub dropped_frames: usize,
    /// Fraction of the frames in dropped states.
    pub dropped_population: f64,
}

/// Split entropy and the terms it is built from, in units of k_B.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitEntropy {
    /// `soft + hard + coupling`.
    pub total: f64,
    /// Entropy of the soft coordinates.
    pub soft: f64,
    /// Entropy of the hard coordinates over all frames.
    pub hard: f64,
    /// Dependence between the sets, at most 0 up to sampling error.
    pub coupling: f64,
    /// Indices of the soft coordinates.
    pub soft_coordinates: Vec<usize>,
    /// Indices of the hard coordinates.
    pub hard_coordinates: Vec<usize>,
    /// Number of basins of each coordinate whose basins were detected.
    pub basin_counts: Vec<Option<usize>>,
    /// The states, for the per-state coupling.
    pub states: Option<StateSummary>,
}

/// Split entropy of the coordinates in `one_d_data`. `options` applies to every
/// part: the metrics mark the periodic coordinates, and the finite-sample
/// constant and Theiler window apply to the nearest-neighbor estimates.
pub fn estimate_split_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    split: &SplitOptions,
    options: &EntropyOptions,
) -> Result<SplitEntropy, String> {
    split_parts(one_d_data, frames_end, split, options, false).map(|(entropy, _)| entropy)
}

/// Each coordinate's share of a split entropy, in units of k_B.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSplitEntropy {
    /// `own + coupling` for each coordinate; sums to `split.total`.
    pub total: Vec<f64>,
    /// A soft coordinate's share of the soft entropy, or a hard coordinate's
    /// of the hard entropy; sums to `split.soft + split.hard`.
    pub own: Vec<f64>,
    /// Each coordinate's share of the coupling term; sums to
    /// `split.coupling`.
    pub coupling: Vec<f64>,
    /// The split entropy the shares divide.
    pub split: SplitEntropy,
}

/// Each coordinate's share of [`estimate_split_entropy`].
///
/// - Soft coordinates take their shares of the soft entropy as
///   [`estimate_coordinate_entropy`] (MIE) or
///   [`estimate_coordinate_mist_entropy`] (MIST) gives them. The joint soft
///   estimate has no such split and is refused.
/// - Hard coordinates take their shares of the QH or BQH entropy as
///   [`estimate_coordinate_quasi_harmonic_entropy`](crate::quasi_harmonic::estimate_coordinate_quasi_harmonic_entropy)
///   gives them: first-order term plus 1/2 (log C)_ii. With
///   [`HardEstimator::QuasiHarmonicMie1`] each takes its first-order term
///   only; with [`HardEstimator::QuasiHarmonicMie2`], each pair's
///   1/2 ln(1 - C_ij^2) is shared equally between its two coordinates instead.
/// - The Gaussian coupling 1/2 [ln|C(S+H)| - ln|C(S)| - ln|C(H)|] is shared
///   the same way: coordinate i receives 1/2 [(log C(S+H))_ii - (log C(S))_ii]
///   if soft, with C(H) if hard, summed over its entries (two per harmonic of
///   a soft torsion). Unlike a QH correlation share, one coordinate's
///   coupling share can be positive; their sum cannot.
/// - The per-state coupling gives each hard coordinate its population-weighted
///   share of the hard entropy within the states, minus its share over all
///   frames, and soft coordinates 0.
///
/// Every set of shares sums to its term up to rounding. The terms in `split`
/// are those of [`estimate_split_entropy`] exactly.
pub fn estimate_coordinate_split_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    split: &SplitOptions,
    options: &EntropyOptions,
) -> Result<CoordinateSplitEntropy, String> {
    let (split, shares) = split_parts(one_d_data, frames_end, split, options, true)?;
    let (own, coupling) = shares.expect("shares were requested");
    Ok(CoordinateSplitEntropy {
        total: own.iter().zip(&coupling).map(|(a, b)| a + b).collect(),
        own,
        coupling,
        split,
    })
}

/// Each coordinate's share of the soft or hard entropy, and of the coupling.
type Shares = (Vec<f64>, Vec<f64>);

/// The split entropy, and with `want_shares` each coordinate's shares of it.
fn split_parts(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    split: &SplitOptions,
    options: &EntropyOptions,
    want_shares: bool,
) -> Result<(SplitEntropy, Option<Shares>), String> {
    let (data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let dimensions = data.len();
    let mut basins: HashMap<usize, Basins> = HashMap::new();
    let mut detect = |index: usize| -> Result<Basins, String> {
        if let Some(found) = basins.get(&index) {
            return Ok(found.clone());
        }
        if !matches!(metrics[index], CoordinateMetric::Periodic { .. }) {
            return Err(format!(
                "coordinate {index} is not periodic, so it has no basins to detect"
            ));
        }
        let found = detect_basins(&data[index], metrics[index], &split.basins)
            .map_err(|err| format!("coordinate {index}: {err}"))?;
        basins.insert(index, found.clone());
        Ok(found)
    };

    let soft_coordinates = match split.soft {
        SoftSelection::Indices(indices) => checked_indices(indices, dimensions, "soft")?,
        SoftSelection::Detect { candidates } => {
            let candidates = match candidates {
                Some(candidates) => checked_indices(candidates, dimensions, "candidate")?,
                None => (0..dimensions)
                    .filter(|&i| matches!(metrics[i], CoordinateMetric::Periodic { .. }))
                    .collect(),
            };
            let mut soft = Vec::new();
            for index in candidates {
                if detect(index)?.is_soft(&split.basins) {
                    soft.push(index);
                }
            }
            soft
        }
    };
    let hard_coordinates: Vec<usize> = (0..dimensions)
        .filter(|i| !soft_coordinates.contains(i))
        .collect();

    let subset_options = |indices: &[usize]| -> Vec<CoordinateMetric> {
        indices.iter().map(|&i| metrics[i]).collect()
    };
    let columns =
        |indices: &[usize]| -> Vec<Vec<f64>> { indices.iter().map(|&i| data[i].clone()).collect() };

    let mut own = vec![0.0; dimensions];
    let mut coupling_shares = vec![0.0; dimensions];

    // Soft entropy.
    let soft = if soft_coordinates.is_empty() {
        0.0
    } else {
        let soft_metrics = subset_options(&soft_coordinates);
        let soft_data = columns(&soft_coordinates);
        let soft = match split.soft_estimator {
            SoftEstimator::NearestNeighborMie { order } => calculate_entropy(
                &soft_data,
                frames_end,
                order,
                &with_metrics(options, &soft_metrics),
            ),
            SoftEstimator::JointNearestNeighbor => estimate_joint_entropy(
                &soft_data,
                frames_end,
                &with_metrics(options, &soft_metrics),
            ),
            SoftEstimator::MutualInformationSpanningTree => estimate_mist_entropy(
                &soft_data,
                frames_end,
                &with_metrics(options, &soft_metrics),
            ),
            SoftEstimator::BayesianNetwork { parents } => estimate_network_entropy(
                &soft_data,
                frames_end,
                parents,
                &with_metrics(options, &soft_metrics),
            ),
        }
        .map_err(|err| format!("soft set: {err}"))?;
        if want_shares {
            let soft_options = with_metrics(options, &soft_metrics);
            let shares = match split.soft_estimator {
                SoftEstimator::NearestNeighborMie { order } => {
                    estimate_coordinate_entropy(&soft_data, frames_end, order, &soft_options)
                }
                SoftEstimator::MutualInformationSpanningTree => {
                    estimate_coordinate_mist_entropy(&soft_data, frames_end, &soft_options)
                }
                SoftEstimator::BayesianNetwork { parents } => estimate_coordinate_network_entropy(
                    &soft_data,
                    frames_end,
                    parents,
                    &soft_options,
                ),
                SoftEstimator::JointNearestNeighbor => Err(
                    "the joint soft estimate has no per-coordinate split; use the MIE, MIST or network"
                        .to_string(),
                ),
            }
            .map_err(|err| format!("soft set: {err}"))?;
            for (&index, share) in soft_coordinates.iter().zip(shares) {
                own[index] = share;
            }
        }
        soft
    };

    // Hard entropy over all frames.
    let hard_metrics = subset_options(&hard_coordinates);
    let hard_data = columns(&hard_coordinates);
    // The hard entropy of some frames, and with `want_shares` each hard
    // coordinate's share of it.
    let hard_entropy = |data: &[Vec<f64>], frames: usize| -> Result<HardParts, String> {
        let hard_options = with_metrics(options, &hard_metrics);
        let truncation = match split.hard_estimator {
            HardEstimator::QuasiHarmonicMie1 => Some(1),
            HardEstimator::QuasiHarmonicMie2 => Some(2),
            _ => None,
        };
        if let Some(order) = truncation {
            let (total, shares) =
                truncated_quasi_harmonic_terms(data, frames, &split.hard, &hard_options, order)?;
            return Ok((total, want_shares.then_some(shares)));
        }
        let boltzmann = split.hard_estimator == HardEstimator::BoltzmannQuasiHarmonic;
        let terms = quasi_harmonic_terms(data, frames, &split.hard, &hard_options, boltzmann)?;
        let total = if boltzmann {
            terms.boltzmann_quasi_harmonic
        } else {
            terms.quasi_harmonic
        };
        // A failed split leaves the total, so the same frames count with and
        // without shares; the callers refuse a missing split.
        let shares = want_shares
            .then(|| {
                coordinate_quasi_harmonic_terms(data, frames, &split.hard, &hard_options, boltzmann)
                    .ok()
            })
            .flatten()
            .map(|terms| {
                if boltzmann {
                    terms.boltzmann_quasi_harmonic
                } else {
                    terms.quasi_harmonic
                }
            });
        Ok((total, shares))
    };
    let (hard, hard_shares) = if hard_coordinates.is_empty() {
        (0.0, None)
    } else {
        hard_entropy(&hard_data, frames_end).map_err(|err| format!("hard set: {err}"))?
    };
    if want_shares && !hard_coordinates.is_empty() {
        let shares = hard_shares.as_ref().ok_or_else(|| {
            "hard set: its entropy could not be split between its coordinates".to_string()
        })?;
        for (&index, &share) in hard_coordinates.iter().zip(shares) {
            own[index] = share;
        }
    }

    // Coupling.
    let mut states = None;
    let coupling = if soft_coordinates.is_empty() || hard_coordinates.is_empty() {
        0.0
    } else {
        match split.coupling {
            Coupling::None => 0.0,
            Coupling::Gaussian { features } => {
                let soft_representation = match features {
                    SoftFeatures::Complex => Representation::Complex,
                    SoftFeatures::CosSin => Representation::Harmonics(1),
                    SoftFeatures::Harmonics(0) => {
                        return Err("soft features need at least one harmonic".to_string());
                    }
                    SoftFeatures::Harmonics(count) => Representation::Harmonics(count),
                };
                let representation = |i: usize| {
                    if soft_coordinates.contains(&i) {
                        soft_representation
                    } else {
                        Representation::of(metrics[i], split.hard.periodic)
                    }
                };
                let inputs = |indices: &[usize]| {
                    let columns: Vec<&[f64]> =
                        indices.iter().map(|&i| data[i].as_slice()).collect();
                    let representations: Vec<Representation> =
                        indices.iter().map(|&i| representation(i)).collect();
                    (columns, subset_options(indices), representations)
                };
                let log_det = |indices: &[usize]| -> Result<f64, String> {
                    let (columns, metrics, representations) = inputs(indices);
                    correlation_log_det(&columns, &metrics, &representations)
                };
                let all: Vec<usize> = (0..dimensions).collect();
                let coupling = 0.5
                    * (log_det(&all).map_err(|err| format!("soft and hard sets: {err}"))?
                        - log_det(&soft_coordinates).map_err(|err| format!("soft set: {err}"))?
                        - log_det(&hard_coordinates).map_err(|err| format!("hard set: {err}"))?);
                if want_shares {
                    // The eigendecomposition behind the shares is stricter about
                    // near-singular matrices than the Cholesky factor behind the
                    // total: higher harmonics of a soft torsion that barely moves
                    // are nearly collinear (sin k phi is about k phi).
                    let log_diagonal = |indices: &[usize]| -> Result<Vec<f64>, String> {
                        let (columns, metrics, representations) = inputs(indices);
                        correlation_log_diagonal(&columns, &metrics, &representations).map_err(
                            |err| {
                                format!(
                                    "coupling shares: {err}; if some soft torsions barely move, their harmonics are nearly collinear: use fewer harmonics or the cos/sin features"
                                )
                            },
                        )
                    };
                    let joint = log_diagonal(&all)?;
                    for set in [&soft_coordinates, &hard_coordinates] {
                        for (&index, alone) in set.iter().zip(log_diagonal(set)?) {
                            coupling_shares[index] = 0.5 * (joint[index] - alone);
                        }
                    }
                }
                coupling
            }
            Coupling::NearestNeighborMie => {
                // Every hard-soft pair, estimated exactly as the soft block's
                // own pair terms are, and subtracted as a pair term is.
                let pairs: Vec<(usize, usize)> = hard_coordinates
                    .iter()
                    .flat_map(|&hard| soft_coordinates.iter().map(move |&soft| (hard, soft)))
                    .collect();
                let values: Vec<f64> = pairs
                    .par_iter()
                    .map(|&(hard, soft)| {
                        let pair = [data[hard].clone(), data[soft].clone()];
                        let pair_metrics = [metrics[hard], metrics[soft]];
                        estimate_coordinate_mutual_information(
                            &pair,
                            frames_end,
                            &with_metrics(options, &pair_metrics),
                        )
                        .map(|mutual| mutual[0])
                    })
                    .collect::<Result<Vec<f64>, String>>()
                    .map_err(|err| format!("hard-soft cross terms: {err}"))?;
                if want_shares {
                    // Each pair term is split between the two coordinates it
                    // involves, as in the rest of the expansion.
                    for (&(hard, soft), &mutual) in pairs.iter().zip(&values) {
                        coupling_shares[hard] -= 0.5 * mutual;
                        coupling_shares[soft] -= 0.5 * mutual;
                    }
                }
                -values.iter().sum::<f64>()
            }
            Coupling::PerState(state_options) => {
                if options.theiler_window > 0
                    && split.hard_estimator == HardEstimator::BoltzmannQuasiHarmonic
                    && split.hard.marginals == MarginalEstimator::NearestNeighbor
                {
                    return Err("per-state coupling cannot apply a Theiler window to the hard set's nearest-neighbor marginals: a state's frames are not consecutive".to_string());
                }
                let (defining, labels) = match state_options.definition {
                    StateDefinition::Labels(labels) => {
                        if labels.len() < frames_end {
                            return Err(format!(
                                "{} state labels for {frames_end} frames",
                                labels.len()
                            ));
                        }
                        (
                            Vec::new(),
                            labels[..frames_end].iter().map(|&l| vec![l]).collect(),
                        )
                    }
                    StateDefinition::Detected { defining, exclude } => {
                        let candidates = match defining {
                            Some(defining) => {
                                checked_indices(defining, dimensions, "state-defining")?
                            }
                            None => soft_coordinates.clone(),
                        };
                        let mut defining = Vec::new();
                        for index in candidates.into_iter().filter(|i| !exclude.contains(i)) {
                            if detect(index)?.count >= 2 {
                                defining.push(index);
                            }
                        }
                        let labels: Vec<Vec<u32>> = (0..frames_end)
                            .map(|frame| defining.iter().map(|i| basins[i].labels[frame]).collect())
                            .collect();
                        (defining, labels)
                    }
                };
                let (conditional, summary, conditional_shares) = per_state(
                    &hard_data,
                    &labels,
                    (hard, hard_shares.as_deref()),
                    &state_options,
                    hard_coordinates.len(),
                    &hard_entropy,
                    defining,
                )?;
                states = Some(summary);
                if let (Some(conditional), Some(over_all)) = (conditional_shares, &hard_shares) {
                    for ((&index, within), all) in
                        hard_coordinates.iter().zip(conditional).zip(over_all)
                    {
                        coupling_shares[index] = within - all;
                    }
                }
                conditional - hard
            }
        }
    };

    let basin_counts = (0..dimensions)
        .map(|i| basins.get(&i).map(|b| b.count))
        .collect();
    let entropy = SplitEntropy {
        total: soft + hard + coupling,
        soft,
        hard,
        coupling,
        soft_coordinates,
        hard_coordinates,
        basin_counts,
        states,
    };
    Ok((entropy, want_shares.then_some((own, coupling_shares))))
}

/// A hard entropy, and each hard coordinate's share of it when asked for.
type HardParts = (f64, Option<Vec<f64>>);

/// The hard entropy of some frames' hard columns: (columns, frame count).
type HardEntropy<'a> = dyn Fn(&[Vec<f64>], usize) -> Result<HardParts, String> + 'a;

/// The population-weighted hard entropy over the states, the states, and the
/// population-weighted shares of each hard coordinate when `hard_over_all_frames`
/// has shares.
fn per_state(
    hard_data: &[Vec<f64>],
    labels: &[Vec<u32>],
    hard_over_all_frames: (f64, Option<&[f64]>),
    options: &StateOptions,
    hard_dimensions: usize,
    hard_entropy: &HardEntropy,
    defining: Vec<usize>,
) -> Result<(f64, StateSummary, Option<Vec<f64>>), String> {
    let (hard_over_all_frames, shares_over_all_frames) = hard_over_all_frames;
    let frames_end = labels.len();
    let mut groups: BTreeMap<&[u32], Vec<usize>> = BTreeMap::new();
    for (frame, label) in labels.iter().enumerate() {
        groups.entry(label.as_slice()).or_default().push(frame);
    }
    if groups.len() == 1 {
        // One state: conditioning changes nothing.
        let state = StateEntropy {
            basins: groups.keys().next().map(|k| k.to_vec()).unwrap_or_default(),
            frames: frames_end,
            population: 1.0,
            hard_entropy: Some(hard_over_all_frames),
        };
        return Ok((
            hard_over_all_frames,
            StateSummary {
                defining,
                states: vec![state],
                dropped_frames: 0,
                dropped_population: 0.0,
            },
            shares_over_all_frames.map(<[f64]>::to_vec),
        ));
    }
    let large = groups
        .values()
        .filter(|frames| frames.len() >= options.min_frames)
        .count();
    if large > options.max_states {
        return Err(format!(
            "{large} conformational states have at least {} frames, more than the maximum of {}",
            options.min_frames, options.max_states
        ));
    }

    let threshold = options.min_frames.max(hard_dimensions + 2);
    let mut states = Vec::with_capacity(groups.len());
    let (mut weighted, mut used) = (0.0, 0.0);
    let mut weighted_shares = shares_over_all_frames.map(|shares| vec![0.0; shares.len()]);
    for (basins, frames) in groups {
        let population = frames.len() as f64 / frames_end as f64;
        let entropy = if frames.len() < threshold {
            None
        } else {
            let gathered: Vec<Vec<f64>> = hard_data
                .iter()
                .map(|column| frames.iter().map(|&f| column[f]).collect())
                .collect();
            hard_entropy(&gathered, frames.len())
                .ok()
                .filter(|(e, _)| e.is_finite())
        };
        if let Some((_, shares)) = &entropy
            && let Some(weighted_shares) = weighted_shares.as_mut()
        {
            let shares = shares.as_ref().ok_or_else(|| {
                format!(
                    "the hard entropy of state {basins:?} could not be split between its coordinates"
                )
            })?;
            for (sum, share) in weighted_shares.iter_mut().zip(shares) {
                *sum += population * share;
            }
        }
        let entropy = entropy.map(|(e, _)| e);
        if let Some(entropy) = entropy {
            weighted += population * entropy;
            used += population;
        }
        states.push(StateEntropy {
            basins: basins.to_vec(),
            frames: frames.len(),
            population,
            hard_entropy: entropy,
        });
    }
    if used == 0.0 {
        return Err(format!(
            "no conformational state has enough frames (at least {threshold}) for its hard entropy"
        ));
    }
    let dropped_frames = states
        .iter()
        .filter(|s| s.hard_entropy.is_none())
        .map(|s| s.frames)
        .sum();
    Ok((
        weighted / used,
        StateSummary {
            defining,
            states,
            dropped_frames,
            dropped_population: 1.0 - used,
        },
        weighted_shares.map(|shares| shares.into_iter().map(|s| s / used).collect()),
    ))
}

/// `options` with the metrics of a subset of the coordinates.
fn with_metrics<'a>(
    options: &EntropyOptions,
    metrics: &'a [CoordinateMetric],
) -> EntropyOptions<'a> {
    EntropyOptions {
        metrics: Some(metrics),
        constant: options.constant,
        theiler_window: options.theiler_window,
    }
}

/// `indices` checked to be in range and distinct, in ascending order.
fn checked_indices(indices: &[usize], dimensions: usize, what: &str) -> Result<Vec<usize>, String> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    if let Some(&bad) = sorted.iter().find(|&&i| i >= dimensions) {
        return Err(format!(
            "{what} coordinate {bad} is out of range for {dimensions} coordinates"
        ));
    }
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(format!("{what} coordinates are repeated"));
    }
    Ok(sorted)
}
