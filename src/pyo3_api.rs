//! The `molentropy` Python extension module, built with maturin.
//!
//! Its functions are called from Python, not from Rust; the user guide
//! documents the Python API: <https://helmutcarter.github.io/molentropy/python.html>. Every function wraps an estimator of this crate and releases
//! the GIL while it runs.

#![allow(unsafe_op_in_unsafe_fn)]
use crate::basins::{BasinOptions, detect_basins as detect_basins_impl};
use crate::bat_library::{BondSet, CoordinateKind, CoordinateOptions, InternalCoordinates};
use crate::bias::BiasCorrection;
use crate::quasi_harmonic::{MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions};
use crate::split::{
    Coupling, DEFAULT_HARMONICS, HardEstimator, LARGE_SOFT_SET, SoftEstimator, SoftFeatures,
    SoftSelection, SplitEntropy, SplitOptions, StateDefinition, StateOptions,
    estimate_coordinate_split_entropy, estimate_split_entropy as split_impl,
    large_soft_set_warning,
};
use crate::{CoordinateMetric, EntropyOptions, FiniteSampleConstant};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::HashMap;
use std::path::Path;

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

fn entropy_options(
    metrics: &[CoordinateMetric],
    exact_constant: bool,
    theiler_window: usize,
) -> EntropyOptions<'_> {
    EntropyOptions {
        metrics: Some(metrics),
        constant: if exact_constant {
            FiniteSampleConstant::Exact
        } else {
            FiniteSampleConstant::Asymptotic
        },
        theiler_window,
    }
}

/// QH and BQH entropy as a dict of the totals and the terms they are built
/// from. `histogram_bins` switches the BQH marginals from nearest neighbor to a
/// histogram.
fn quasi_harmonic(
    one_d_data: &[Vec<f64>],
    frames: usize,
    metrics: &[CoordinateMetric],
    histogram_bins: Option<usize>,
    periodic_correlation: &str,
    bias: &str,
    effective_frames: Option<f64>,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<HashMap<&'static str, f64>> {
    let entropy = crate::quasi_harmonic::estimate_quasi_harmonic_entropy(
        one_d_data,
        frames,
        &quasi_harmonic_options(histogram_bins, periodic_correlation, bias, effective_frames)?,
        &entropy_options(metrics, exact_constant, theiler_window),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(HashMap::from([
        ("quasi_harmonic", entropy.quasi_harmonic),
        ("boltzmann_quasi_harmonic", entropy.boltzmann_quasi_harmonic),
        ("first_order_gaussian", entropy.first_order_gaussian),
        ("first_order_boltzmann", entropy.first_order_boltzmann),
        ("correlation", entropy.correlation),
    ]))
}

/// Each coordinate's share of the QH and BQH entropy, as a dict of lists with
/// the same keys as `quasi_harmonic`.
fn coordinate_quasi_harmonic(
    one_d_data: &[Vec<f64>],
    frames: usize,
    metrics: &[CoordinateMetric],
    histogram_bins: Option<usize>,
    periodic_correlation: &str,
    bias: &str,
    effective_frames: Option<f64>,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<HashMap<&'static str, Vec<f64>>> {
    let entropy = crate::quasi_harmonic::estimate_coordinate_quasi_harmonic_entropy(
        one_d_data,
        frames,
        &quasi_harmonic_options(histogram_bins, periodic_correlation, bias, effective_frames)?,
        &entropy_options(metrics, exact_constant, theiler_window),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(HashMap::from([
        ("quasi_harmonic", entropy.quasi_harmonic),
        ("boltzmann_quasi_harmonic", entropy.boltzmann_quasi_harmonic),
        ("first_order_gaussian", entropy.first_order_gaussian),
        ("first_order_boltzmann", entropy.first_order_boltzmann),
        ("correlation", entropy.correlation),
    ]))
}

fn quasi_harmonic_options(
    histogram_bins: Option<usize>,
    periodic_correlation: &str,
    bias: &str,
    effective_frames: Option<f64>,
) -> PyResult<QuasiHarmonicOptions> {
    let bias = match bias {
        "none" => BiasCorrection::None,
        "wishart" => BiasCorrection::Wishart,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bias must be \"none\" or \"wishart\", got {other:?}"
            )));
        }
    };
    Ok(QuasiHarmonicOptions {
        bias,
        effective_frames,
        marginals: match histogram_bins {
            Some(bins) => MarginalEstimator::Histogram { bins },
            None => MarginalEstimator::NearestNeighbor,
        },
        periodic: match periodic_correlation {
            "complex" => PeriodicCorrelation::Complex,
            "centered" => PeriodicCorrelation::Centered,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "periodic_correlation must be 'complex' or 'centered', got {other:?}"
                )));
            }
        },
    })
}

fn array_to_one_d_data(data: PyReadonlyArray2<f64>) -> (Vec<Vec<f64>>, usize) {
    let array = data.as_array();
    let one_d_data: Vec<Vec<f64>> = array.outer_iter().map(|row| row.to_vec()).collect();
    let frames_end = array.shape()[1];
    (one_d_data, frames_end)
}

fn metrics_from_periods(
    periods: Option<Vec<Option<f64>>>,
    dimensions: usize,
) -> PyResult<Vec<CoordinateMetric>> {
    let Some(periods) = periods else {
        return Ok(vec![CoordinateMetric::Linear; dimensions]);
    };
    if periods.len() != dimensions {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "received {} periods for {dimensions} coordinates",
            periods.len()
        )));
    }
    periods
        .into_iter()
        .enumerate()
        .map(|(index, period)| match period {
            None => Ok(CoordinateMetric::Linear),
            Some(period) if period.is_finite() && period > 0.0 => {
                Ok(CoordinateMetric::Periodic { period })
            }
            Some(_) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "period for coordinate {index} must be finite and positive"
            ))),
        })
        .collect()
}

fn system_from_trajectory(
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
    options: CoordinateOptions,
) -> PyResult<System> {
    let start = start.unwrap_or(0);
    let stride = stride.unwrap_or(1);
    let mut internal =
        InternalCoordinates::with_options(Path::new(top_path), options).map_err(to_py_err)?;
    internal
        .calculate_internal_coords(
            Path::new(traj_path),
            stop.unwrap_or(usize::MAX),
            torsions_only.unwrap_or(false),
        )
        .map_err(to_py_err)?;
    let one_d_data = internal
        .columns(start, stride)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let metrics = internal.coordinate_metrics();

    // The Jacobian is only defined for the tree bond set with every coordinate;
    // otherwise the System has none to report.
    let mean_log_jacobian = internal.log_jacobian().ok().map(|log_jacobian| {
        let selected: Vec<f64> = log_jacobian
            .into_iter()
            .skip(start)
            .step_by(stride)
            .collect();
        selected.iter().sum::<f64>() / selected.len() as f64
    });

    let frames = one_d_data[0].len();
    Ok(System {
        one_d_data,
        frames,
        metrics,
        mean_log_jacobian,
        atoms: internal.coordinate_atoms(),
        kinds: internal.coordinate_kinds(),
        soft_torsions: internal.soft_torsions(),
        symmetric_rotor_torsions: internal.symmetric_rotor_torsions(),
        ring_torsions: internal.ring_torsions(),
    })
}

/// The `soft` argument of `estimate_split_entropy`.
enum SoftArgument {
    /// None: topology for a System, detection for an array.
    Default,
    Topology,
    Detect,
    List(Vec<usize>),
}

impl SoftArgument {
    fn extract(soft: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(soft) = soft.filter(|soft| !soft.is_none()) else {
            return Ok(Self::Default);
        };
        if let Ok(name) = soft.extract::<String>() {
            return match name.as_str() {
                "topology" => Ok(Self::Topology),
                "detect" => Ok(Self::Detect),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "soft must be None, 'topology', 'detect' or a list of coordinate indices, got {other:?}"
                ))),
            };
        }
        Ok(Self::List(soft.extract()?))
    }
}

/// The keyword arguments of `estimate_split_entropy`, checked.
struct SplitArguments {
    soft: SoftArgument,
    soft_estimator: SoftEstimator,
    hard_estimator: HardEstimator,
    hard: QuasiHarmonicOptions,
    coupling: String,
    soft_features: SoftFeatures,
    state_coordinates: Option<Vec<usize>>,
    state_labels: Option<Vec<u32>>,
    min_state_frames: usize,
}

impl SplitArguments {
    #[allow(clippy::too_many_arguments)]
    fn new(
        soft: Option<&Bound<'_, PyAny>>,
        soft_estimator: &str,
        mie_order: usize,
        hard_estimator: &str,
        coupling: &str,
        soft_features: &str,
        harmonics: Option<usize>,
        state_coordinates: Option<Vec<usize>>,
        state_labels: Option<Vec<u32>>,
        min_state_frames: usize,
        histogram_bins: Option<usize>,
        periodic_correlation: &str,
        bias: &str,
        effective_frames: Option<f64>,
        network_parents: usize,
    ) -> PyResult<Self> {
        let invalid = |name: &str, allowed: &str, got: &str| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must be {allowed}, got {got:?}"
            ))
        };
        let soft_estimator = match soft_estimator {
            "mie" => SoftEstimator::NearestNeighborMie { order: mie_order },
            "mist" => SoftEstimator::MutualInformationSpanningTree,
            "network" => SoftEstimator::BayesianNetwork {
                parents: network_parents,
            },
            "joint" => SoftEstimator::JointNearestNeighbor,
            other => {
                return Err(invalid(
                    "soft_estimator",
                    "'mie', 'mist', 'network' or 'joint'",
                    other,
                ));
            }
        };
        let hard_estimator = match hard_estimator {
            "qh" => HardEstimator::QuasiHarmonic,
            "bqh" => HardEstimator::BoltzmannQuasiHarmonic,
            "qh-mie1" => HardEstimator::QuasiHarmonicMie1,
            "qh-mie2" => HardEstimator::QuasiHarmonicMie2,
            other => {
                return Err(invalid(
                    "hard_estimator",
                    "'qh', 'bqh', 'qh-mie1' or 'qh-mie2'",
                    other,
                ));
            }
        };
        if !["gaussian", "states", "none"].contains(&coupling) {
            return Err(invalid(
                "coupling",
                "'gaussian', 'states' or 'none'",
                coupling,
            ));
        }
        if harmonics.is_some() && soft_features != "harmonics" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "harmonics applies only to soft_features='harmonics'",
            ));
        }
        let soft_features = match soft_features {
            "complex" => SoftFeatures::Complex,
            "cossin" => SoftFeatures::CosSin,
            "harmonics" if harmonics == Some(0) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "harmonics must be at least 1",
                ));
            }
            "harmonics" => SoftFeatures::Harmonics(harmonics.unwrap_or(DEFAULT_HARMONICS)),
            other => {
                return Err(invalid(
                    "soft_features",
                    "'complex', 'cossin' or 'harmonics'",
                    other,
                ));
            }
        };
        if coupling != "states" && (state_coordinates.is_some() || state_labels.is_some()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "state_coordinates and state_labels apply only to coupling='states'",
            ));
        }
        if state_coordinates.is_some() && state_labels.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pass state_coordinates or state_labels, not both",
            ));
        }
        Ok(Self {
            soft: SoftArgument::extract(soft)?,
            soft_estimator,
            hard_estimator,
            hard: quasi_harmonic_options(
                histogram_bins,
                periodic_correlation,
                bias,
                effective_frames,
            )?,
            coupling: coupling.to_string(),
            soft_features,
            state_coordinates,
            state_labels,
            min_state_frames,
        })
    }

    /// The split entropy of `data`. `topology` is a System's soft torsions and
    /// symmetric rotors; None for an array.
    #[allow(clippy::too_many_arguments)]
    fn estimate(
        &self,
        data: &[Vec<f64>],
        frames: usize,
        metrics: &[CoordinateMetric],
        topology: Option<(&[usize], &[usize])>,
        exact_constant: bool,
        theiler_window: usize,
        per_coordinate: bool,
    ) -> PyResult<(SplitEntropy, Option<CoordinateShares>)> {
        let selection = match (&self.soft, topology) {
            (SoftArgument::List(list), _) => SoftSelection::Indices(list),
            (SoftArgument::Detect, _) | (SoftArgument::Default, None) => {
                SoftSelection::Detect { candidates: None }
            }
            (SoftArgument::Topology | SoftArgument::Default, Some((soft, _))) => {
                SoftSelection::Indices(soft)
            }
            (SoftArgument::Topology, None) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "soft='topology' needs a System; for an array, pass 'detect' or a list",
                ));
            }
        };
        let symmetric: &[usize] = topology.map_or(&[], |(_, symmetric)| symmetric);
        let mut split = SplitOptions::new(selection);
        split.soft_estimator = self.soft_estimator;
        split.hard_estimator = self.hard_estimator;
        split.hard = self.hard;
        split.coupling = match self.coupling.as_str() {
            "none" => Coupling::None,
            "gaussian" => Coupling::Gaussian {
                features: self.soft_features,
            },
            _ => Coupling::PerState(StateOptions {
                definition: match (&self.state_labels, &self.state_coordinates) {
                    (Some(labels), _) => StateDefinition::Labels(labels),
                    (None, Some(defining)) => StateDefinition::Detected {
                        defining: Some(defining),
                        exclude: &[],
                    },
                    (None, None) => StateDefinition::Detected {
                        defining: None,
                        exclude: symmetric,
                    },
                },
                min_frames: self.min_state_frames,
                ..Default::default()
            }),
        };
        let options = entropy_options(metrics, exact_constant, theiler_window);
        if per_coordinate {
            estimate_coordinate_split_entropy(data, frames, &split, &options).map(|shares| {
                let parts = (shares.total, shares.own, shares.coupling);
                (shares.split, Some(parts))
            })
        } else {
            split_impl(data, frames, &split, &options).map(|result| (result, None))
        }
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

/// `SplitEntropy` as a dict.
/// Each coordinate's total, own and coupling shares of a split entropy.
type CoordinateShares = (Vec<f64>, Vec<f64>, Vec<f64>);

fn split_to_dict(
    py: Python<'_>,
    (result, shares): (SplitEntropy, Option<CoordinateShares>),
) -> PyResult<PyObject> {
    if result.soft_coordinates.len() > LARGE_SOFT_SET {
        PyErr::warn_bound(
            py,
            &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
            &large_soft_set_warning(result.soft_coordinates.len()),
            1,
        )?;
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("total", result.total)?;
    dict.set_item("soft", result.soft)?;
    dict.set_item("hard", result.hard)?;
    dict.set_item("coupling", result.coupling)?;
    dict.set_item("soft_coordinates", result.soft_coordinates)?;
    dict.set_item("hard_coordinates", result.hard_coordinates)?;
    dict.set_item("basin_counts", result.basin_counts)?;
    match result.states {
        None => dict.set_item("states", py.None())?,
        Some(summary) => {
            let states = PyDict::new_bound(py);
            states.set_item("defining", summary.defining)?;
            states.set_item("dropped_frames", summary.dropped_frames)?;
            states.set_item("dropped_population", summary.dropped_population)?;
            let list: Vec<PyObject> = summary
                .states
                .into_iter()
                .map(|state| -> PyResult<PyObject> {
                    let entry = PyDict::new_bound(py);
                    entry.set_item("basins", state.basins)?;
                    entry.set_item("frames", state.frames)?;
                    entry.set_item("population", state.population)?;
                    entry.set_item("hard_entropy", state.hard_entropy)?;
                    Ok(entry.into_py(py))
                })
                .collect::<PyResult<_>>()?;
            states.set_item("states", list)?;
            dict.set_item("states", states)?;
        }
    }
    if let Some((total, own, coupling)) = shares {
        let coordinates = PyDict::new_bound(py);
        coordinates.set_item("total", total)?;
        coordinates.set_item("own", own)?;
        coordinates.set_item("coupling", coupling)?;
        dict.set_item("coordinates", coordinates)?;
    }
    Ok(dict.into_py(py))
}

/// The data, frame count and metrics of a System or an array. `periodic=False`
/// measures every coordinate linearly, for this calculation only.
fn one_d_data_from_py(
    data: &Bound<'_, PyAny>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
) -> PyResult<(Vec<Vec<f64>>, usize, Vec<CoordinateMetric>)> {
    if let Ok(system) = data.extract::<PyRef<'_, System>>() {
        if periods.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "periods cannot override metrics stored in a System",
            ));
        }
        return Ok((
            system.one_d_data.clone(),
            system.frames,
            system.metrics_for(periodic),
        ));
    }

    if periods.is_some() && !periodic {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods and periodic=False contradict each other; pass one",
        ));
    }
    let array = data.extract::<PyReadonlyArray2<f64>>()?;
    let (one_d_data, frames) = array_to_one_d_data(array);
    let metrics = metrics_from_periods(periods, one_d_data.len())?;
    Ok((one_d_data, frames, metrics))
}

#[pyclass]
struct System {
    one_d_data: Vec<Vec<f64>>,
    frames: usize,
    metrics: Vec<CoordinateMetric>,
    mean_log_jacobian: Option<f64>,
    atoms: Vec<Vec<usize>>,
    kinds: Vec<CoordinateKind>,
    soft_torsions: Vec<usize>,
    symmetric_rotor_torsions: Vec<usize>,
    ring_torsions: Vec<usize>,
}

impl System {
    /// The stored metrics, or every coordinate linear when `periodic` is
    /// false.
    fn metrics_for(&self, periodic: bool) -> Vec<CoordinateMetric> {
        if periodic {
            self.metrics.clone()
        } else {
            vec![CoordinateMetric::Linear; self.metrics.len()]
        }
    }
}

#[pymethods]
impl System {
    #[getter]
    fn frames(&self) -> usize {
        self.frames
    }

    #[getter]
    fn coordinates(&self) -> usize {
        self.one_d_data.len()
    }

    /// Topology atom indices of each coordinate, counted from 0: two for a
    /// bond, three for an angle (vertex in the middle), four for a torsion.
    #[getter]
    fn atoms(&self) -> Vec<Vec<usize>> {
        self.atoms.clone()
    }

    /// The kind of each coordinate: "bond", "angle", "torsion" or "phase angle".
    #[getter]
    fn kinds(&self) -> Vec<&'static str> {
        self.kinds.iter().map(|kind| kind.name()).collect()
    }

    /// For each coordinate, the index of the torsion it is a phase angle of,
    /// or None.
    #[getter]
    fn phase_references(&self) -> Vec<Option<usize>> {
        self.kinds
            .iter()
            .map(|kind| match kind {
                CoordinateKind::PhaseAngle { reference } => Some(*reference),
                _ => None,
            })
            .collect()
    }

    /// The primary torsion about each rotatable (non-ring) bond: the split
    /// estimator's default soft set.
    #[getter]
    fn soft_torsions(&self) -> Vec<usize> {
        self.soft_torsions.clone()
    }

    /// The soft torsions about symmetric rotors such as methyl groups.
    #[getter]
    fn symmetric_rotor_torsions(&self) -> Vec<usize> {
        self.symmetric_rotor_torsions.clone()
    }

    /// Torsions about ring bonds.
    #[getter]
    fn ring_torsions(&self) -> Vec<usize> {
        self.ring_torsions.clone()
    }

    /// The internal coordinates of the selected frames, one list per
    /// coordinate, in the order of `atoms`, `kinds` and `periods`.
    #[getter]
    fn internal_coordinates(&self) -> Vec<Vec<f64>> {
        self.one_d_data.clone()
    }

    #[getter]
    fn periods(&self) -> Vec<Option<f64>> {
        self.metrics
            .iter()
            .map(|metric| match metric {
                CoordinateMetric::Linear => None,
                CoordinateMetric::Periodic { period } => Some(*period),
            })
            .collect()
    }

    /// Mean over the selected frames of ln J, the internal part of the BAT
    /// Jacobian. Available only for a System loaded with `tree_bonds=True`
    /// (or `bonds='tree-unconstrained'`) and without `torsions_only`.
    #[getter]
    fn mean_log_jacobian(&self) -> PyResult<f64> {
        self.mean_log_jacobian.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "the BAT Jacobian needs tree_bonds=True and every coordinate type",
            )
        })
    }

    #[pyo3(signature = (mie_order=None, periodic=true, exact_constant=false, theiler_window=0))]
    fn estimate_entropy(
        &self,
        py: Python<'_>,
        mie_order: Option<usize>,
        periodic: bool,
        exact_constant: bool,
        theiler_window: usize,
    ) -> PyResult<f64> {
        py.allow_threads(|| {
            crate::calculate_entropy(
                &self.one_d_data,
                self.frames,
                mie_order.unwrap_or(2),
                &entropy_options(&self.metrics_for(periodic), exact_constant, theiler_window),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    #[pyo3(signature = (mie_order=None, periodic=true, exact_constant=false, theiler_window=0))]
    fn estimate_coordinate_entropy(
        &self,
        py: Python<'_>,
        mie_order: Option<usize>,
        periodic: bool,
        exact_constant: bool,
        theiler_window: usize,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            crate::estimate_coordinate_entropy(
                &self.one_d_data,
                self.frames,
                mie_order.unwrap_or(2),
                &entropy_options(&self.metrics_for(periodic), exact_constant, theiler_window),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    /// QH and BQH entropy; see the module function of the same name.
    #[pyo3(signature = (histogram_bins=None, periodic_correlation="complex", bias="none", effective_frames=None, periodic=true, exact_constant=false, theiler_window=0))]
    fn estimate_quasi_harmonic_entropy(
        &self,
        py: Python<'_>,
        histogram_bins: Option<usize>,
        periodic_correlation: &str,
        bias: &str,
        effective_frames: Option<f64>,
        periodic: bool,
        exact_constant: bool,
        theiler_window: usize,
    ) -> PyResult<HashMap<&'static str, f64>> {
        py.allow_threads(|| {
            quasi_harmonic(
                &self.one_d_data,
                self.frames,
                &self.metrics_for(periodic),
                histogram_bins,
                periodic_correlation,
                bias,
                effective_frames,
                exact_constant,
                theiler_window,
            )
        })
    }

    /// Each coordinate's share of the QH and BQH entropy; see the module
    /// function of the same name.
    #[pyo3(signature = (histogram_bins=None, periodic_correlation="complex", bias="none", effective_frames=None, periodic=true, exact_constant=false, theiler_window=0))]
    fn estimate_coordinate_quasi_harmonic_entropy(
        &self,
        py: Python<'_>,
        histogram_bins: Option<usize>,
        periodic_correlation: &str,
        bias: &str,
        effective_frames: Option<f64>,
        periodic: bool,
        exact_constant: bool,
        theiler_window: usize,
    ) -> PyResult<HashMap<&'static str, Vec<f64>>> {
        py.allow_threads(|| {
            coordinate_quasi_harmonic(
                &self.one_d_data,
                self.frames,
                &self.metrics_for(periodic),
                histogram_bins,
                periodic_correlation,
                bias,
                effective_frames,
                exact_constant,
                theiler_window,
            )
        })
    }

    /// Split entropy; see the module function of the same name. `soft=None`
    /// uses the System's soft torsions.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (soft=None, soft_estimator="mie", mie_order=2, hard_estimator="qh", coupling="gaussian", soft_features="harmonics", harmonics=None, state_coordinates=None, state_labels=None, min_state_frames=300, histogram_bins=None, periodic_correlation="centered", bias="none", effective_frames=None, periodic=true, exact_constant=false, theiler_window=0, per_coordinate=false, network_parents=5))]
    fn estimate_split_entropy(
        &self,
        py: Python<'_>,
        soft: Option<&Bound<'_, PyAny>>,
        soft_estimator: &str,
        mie_order: usize,
        hard_estimator: &str,
        coupling: &str,
        soft_features: &str,
        harmonics: Option<usize>,
        state_coordinates: Option<Vec<usize>>,
        state_labels: Option<Vec<u32>>,
        min_state_frames: usize,
        histogram_bins: Option<usize>,
        periodic_correlation: &str,
        bias: &str,
        effective_frames: Option<f64>,
        periodic: bool,
        exact_constant: bool,
        theiler_window: usize,
        per_coordinate: bool,
        network_parents: usize,
    ) -> PyResult<PyObject> {
        let arguments = SplitArguments::new(
            soft,
            soft_estimator,
            mie_order,
            hard_estimator,
            coupling,
            soft_features,
            harmonics,
            state_coordinates,
            state_labels,
            min_state_frames,
            histogram_bins,
            periodic_correlation,
            bias,
            effective_frames,
            network_parents,
        )?;
        let metrics = self.metrics_for(periodic);
        let result = py.allow_threads(|| {
            arguments.estimate(
                &self.one_d_data,
                self.frames,
                &metrics,
                Some((&self.soft_torsions, &self.symmetric_rotor_torsions)),
                exact_constant,
                theiler_window,
                per_coordinate,
            )
        })?;
        split_to_dict(py, result)
    }

    #[pyo3(signature = (periodic=true, exact_constant=false, theiler_window=0))]
    fn estimate_coordinate_mutual_information(
        &self,
        py: Python<'_>,
        periodic: bool,
        exact_constant: bool,
        theiler_window: usize,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            crate::estimate_coordinate_mutual_information(
                &self.one_d_data,
                self.frames,
                &entropy_options(&self.metrics_for(periodic), exact_constant, theiler_window),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }
}

// Python keyword arguments, so the count is set by the public signature.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (top_path, traj_path, start=None, stop=None, torsions_only=None, stride=None, tree_bonds=false, phase_angles=true, bonds=None))]
fn load_system(
    py: Python<'_>,
    top_path: &str,
    traj_path: &str,
    start: Option<usize>,
    stop: Option<usize>,
    torsions_only: Option<bool>,
    stride: Option<usize>,
    tree_bonds: bool,
    phase_angles: bool,
    bonds: Option<&str>,
) -> PyResult<System> {
    let bonds = match bonds {
        None if tree_bonds => BondSet::Tree,
        None => BondSet::HeavyAtom,
        Some(_) if tree_bonds => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tree_bonds=True is shorthand for bonds='tree'; pass one or the other",
            ));
        }
        Some("heavy") => BondSet::HeavyAtom,
        Some("tree") => BondSet::Tree,
        Some("tree-unconstrained") => BondSet::TreeUnconstrained,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bonds must be 'heavy', 'tree' or 'tree-unconstrained', got {other:?}"
            )));
        }
    };
    let options = CoordinateOptions {
        bonds,
        phase_angles,
        ..Default::default()
    };
    py.allow_threads(move || {
        system_from_trajectory(
            top_path,
            traj_path,
            start,
            stop,
            torsions_only,
            stride,
            options,
        )
    })
}

/// Total entropy from a mutual information expansion of order `mie_order`
/// (default 2; order 1 is the sum of marginal entropies).
#[pyfunction(signature = (data, mie_order=None, periods=None, periodic=true, exact_constant=false, theiler_window=0))]
fn estimate_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    mie_order: Option<usize>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<f64> {
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods, periodic)?;
    py.allow_threads(move || {
        crate::calculate_entropy(
            &one_d_data,
            frames_end,
            mie_order.unwrap_or(2),
            &entropy_options(&metrics, exact_constant, theiler_window),
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

/// Each coordinate's share of the order-`mie_order` entropy (default 2). The
/// values sum to `estimate_entropy` at the same order; order 1 gives the
/// marginal entropies.
#[pyfunction(signature = (data, mie_order=None, periods=None, periodic=true, exact_constant=false, theiler_window=0))]
fn estimate_coordinate_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    mie_order: Option<usize>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods, periodic)?;
    py.allow_threads(move || {
        crate::estimate_coordinate_entropy(
            &one_d_data,
            frames_end,
            mie_order.unwrap_or(2),
            &entropy_options(&metrics, exact_constant, theiler_window),
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

// Create python wrapper to take BAT coordinates and return a numpy array with the mutual information for each coordinate pair
#[pyfunction(signature = (data, periods=None, periodic=true, exact_constant=false, theiler_window=0))]
fn estimate_coordinate_mutual_information(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<Vec<f64>> {
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods, periodic)?;
    py.allow_threads(move || {
        crate::estimate_coordinate_mutual_information(
            &one_d_data,
            frames_end,
            &entropy_options(&metrics, exact_constant, theiler_window),
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

/// Quasi-harmonic (QH) and Boltzmann quasi-harmonic (BQH) entropy, as a dict
/// with keys `quasi_harmonic`, `boltzmann_quasi_harmonic`,
/// `first_order_gaussian`, `first_order_boltzmann` and `correlation`. BQH
/// uses nearest-neighbor marginals unless `histogram_bins` is given. Periodic
/// coordinates enter the correlation matrix as e^(i phi) unless
/// `periodic_correlation="centered"`.
// Python keyword arguments, so the count is set by the public signature.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (data, periods=None, periodic=true, histogram_bins=None, periodic_correlation="complex", bias="none", effective_frames=None, exact_constant=false, theiler_window=0))]
fn estimate_quasi_harmonic_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
    histogram_bins: Option<usize>,
    periodic_correlation: &str,
    bias: &str,
    effective_frames: Option<f64>,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<HashMap<&'static str, f64>> {
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods, periodic)?;
    py.allow_threads(move || {
        quasi_harmonic(
            &one_d_data,
            frames_end,
            &metrics,
            histogram_bins,
            periodic_correlation,
            bias,
            effective_frames,
            exact_constant,
            theiler_window,
        )
    })
}

/// Each coordinate's share of the QH and BQH entropy, as a dict of lists with
/// the keys of `estimate_quasi_harmonic_entropy`. Every list sums to the
/// matching total; the correlation term is shared out as 1/2 (log C)_ii.
// Python keyword arguments, so the count is set by the public signature.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (data, periods=None, periodic=true, histogram_bins=None, periodic_correlation="complex", bias="none", effective_frames=None, exact_constant=false, theiler_window=0))]
fn estimate_coordinate_quasi_harmonic_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
    histogram_bins: Option<usize>,
    periodic_correlation: &str,
    bias: &str,
    effective_frames: Option<f64>,
    exact_constant: bool,
    theiler_window: usize,
) -> PyResult<HashMap<&'static str, Vec<f64>>> {
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods, periodic)?;
    py.allow_threads(move || {
        coordinate_quasi_harmonic(
            &one_d_data,
            frames_end,
            &metrics,
            histogram_bins,
            periodic_correlation,
            bias,
            effective_frames,
            exact_constant,
            theiler_window,
        )
    })
}

/// Split entropy: soft coordinates by a nearest-neighbor estimate, hard ones by
/// a Gaussian, plus a coupling term. Returns a dict with `total`, `soft`,
/// `hard`, `coupling`, `soft_coordinates`, `hard_coordinates`, `basin_counts`
/// and `states`. `soft` is None, "topology" (a System's soft torsions),
/// "detect", or a list of coordinate indices from 0; None means "topology"
/// for a System and "detect" for an array.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (data, soft=None, soft_estimator="mie", mie_order=2, hard_estimator="qh", coupling="gaussian", soft_features="harmonics", harmonics=None, state_coordinates=None, state_labels=None, min_state_frames=300, histogram_bins=None, periodic_correlation="centered", bias="none", effective_frames=None, periods=None, periodic=true, exact_constant=false, theiler_window=0, per_coordinate=false, network_parents=5))]
fn estimate_split_entropy(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    soft: Option<&Bound<'_, PyAny>>,
    soft_estimator: &str,
    mie_order: usize,
    hard_estimator: &str,
    coupling: &str,
    soft_features: &str,
    harmonics: Option<usize>,
    state_coordinates: Option<Vec<usize>>,
    state_labels: Option<Vec<u32>>,
    min_state_frames: usize,
    histogram_bins: Option<usize>,
    periodic_correlation: &str,
    bias: &str,
    effective_frames: Option<f64>,
    periods: Option<Vec<Option<f64>>>,
    periodic: bool,
    exact_constant: bool,
    theiler_window: usize,
    per_coordinate: bool,
    network_parents: usize,
) -> PyResult<PyObject> {
    if let Ok(system) = data.extract::<PyRef<'_, System>>() {
        if periods.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "periods cannot override metrics stored in a System",
            ));
        }
        return system.estimate_split_entropy(
            py,
            soft,
            soft_estimator,
            mie_order,
            hard_estimator,
            coupling,
            soft_features,
            harmonics,
            state_coordinates,
            state_labels,
            min_state_frames,
            histogram_bins,
            periodic_correlation,
            bias,
            effective_frames,
            periodic,
            exact_constant,
            theiler_window,
            per_coordinate,
            network_parents,
        );
    }
    let arguments = SplitArguments::new(
        soft,
        soft_estimator,
        mie_order,
        hard_estimator,
        coupling,
        soft_features,
        harmonics,
        state_coordinates,
        state_labels,
        min_state_frames,
        histogram_bins,
        periodic_correlation,
        bias,
        effective_frames,
        network_parents,
    )?;
    let (one_d_data, frames_end, metrics) = one_d_data_from_py(data, periods, periodic)?;
    let result = py.allow_threads(|| {
        arguments.estimate(
            &one_d_data,
            frames_end,
            &metrics,
            None,
            exact_constant,
            theiler_window,
            per_coordinate,
        )
    })?;
    split_to_dict(py, result)
}

/// The basins of a periodic coordinate, as a dict with `count`, `labels` (the
/// basin of each value), `boundaries`, `populations`, `resultant` and `soft`
/// (whether the split estimator would call it soft).
// Python keyword arguments, so the count is set by the public signature.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (values, period=std::f64::consts::TAU, bins=72, smoothing=2.0, saddle_fraction=0.5, min_population=0.02, free_rotor_resultant=Some(0.3)))]
fn detect_basins(
    py: Python<'_>,
    values: Vec<f64>,
    period: f64,
    bins: usize,
    smoothing: f64,
    saddle_fraction: f64,
    min_population: f64,
    free_rotor_resultant: Option<f64>,
) -> PyResult<PyObject> {
    let options = BasinOptions {
        bins,
        smoothing,
        saddle_fraction,
        min_population,
        free_rotor_resultant,
    };
    let basins = detect_basins_impl(&values, CoordinateMetric::Periodic { period }, &options)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("soft", basins.is_soft(&options))?;
    dict.set_item("count", basins.count)?;
    dict.set_item("labels", basins.labels)?;
    dict.set_item("boundaries", basins.boundaries)?;
    dict.set_item("populations", basins.populations)?;
    dict.set_item("resultant", basins.resultant)?;
    Ok(dict.into_py(py))
}

#[pymodule]
fn molentropy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<System>()?;
    m.add_function(wrap_pyfunction!(load_system, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_coordinate_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_quasi_harmonic_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_split_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(detect_basins, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimate_coordinate_quasi_harmonic_entropy,
        m
    )?)?;
    Ok(())
}
