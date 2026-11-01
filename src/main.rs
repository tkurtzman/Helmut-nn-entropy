use std::env;
use std::path::Path;

use molentropy::bat_library::{
    BondSet, CoordinateOptions, Geometry, InternalCoordinates, TreeRule,
};
use molentropy::bias::BiasCorrection;
use molentropy::legacy::legacy_entropy;
use molentropy::quasi_harmonic::{
    MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions,
    estimate_coordinate_quasi_harmonic_entropy, estimate_quasi_harmonic_entropy,
};
use molentropy::split::{
    Coupling, HardEstimator, LARGE_SOFT_SET, SoftEstimator, SoftFeatures, SoftSelection,
    SplitOptions, StateDefinition, StateOptions, estimate_coordinate_split_entropy,
    estimate_split_entropy, large_soft_set_warning,
};
use molentropy::{
    CoordinateMetric, DEFAULT_NETWORK_PARENTS, EntropyOptions, FiniteSampleConstant,
    calculate_entropy, estimate_coordinate_entropy, estimate_coordinate_mutual_information,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Estimator {
    NearestNeighbor,
    QuasiHarmonic,
    BoltzmannQuasiHarmonic,
    Split,
}

/// How the split estimator's coupling term is computed, as given on the
/// command line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CouplingFlag {
    Gaussian,
    States,
    None,
}

/// A soft or state-defining set: from topology (or detection), or a list.
#[derive(Clone, PartialEq, Eq)]
enum CoordinateChoice {
    Topology,
    Detect,
    /// Coordinate numbers counted from 1, as printed by --per-coordinate.
    List(Vec<usize>),
}

/// Prints `message` and exits with status 1.
fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// The value of `result`, or `context` and the error, then exit with status 1.
fn or_fail<T>(result: Result<T, String>, context: &str) -> T {
    result.unwrap_or_else(|err| fail(format!("{context}: {err}")))
}

/// The value after the flag `args[i]`, one of `choices`.
fn flag_choice<T: Copy>(args: &[String], i: usize, choices: &[(&str, T)]) -> T {
    let flag = &args[i];
    let Some(raw) = args.get(i + 1) else {
        fail(format!("{flag} requires a value"));
    };
    match choices.iter().find(|(name, _)| name == raw) {
        Some(&(_, value)) => value,
        None => {
            let names: Vec<&str> = choices.iter().map(|(name, _)| *name).collect();
            fail(format!(
                "{flag} must be one of {}, got {raw}",
                names.join(", ")
            ))
        }
    }
}

/// The value after the flag `args[i]`: `keyword` or a comma-separated list
/// of coordinate numbers counted from 1.
fn coordinate_choice(
    args: &[String],
    i: usize,
    keyword: (&str, CoordinateChoice),
) -> CoordinateChoice {
    let flag = &args[i];
    let Some(raw) = args.get(i + 1) else {
        fail(format!("{flag} requires a value"));
    };
    if raw == keyword.0 {
        return keyword.1;
    }
    if raw == "detect" && flag == "--soft" {
        return CoordinateChoice::Detect;
    }
    let numbers: Option<Vec<usize>> = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>().ok().filter(|&n| n >= 1))
        .collect();
    match numbers {
        Some(numbers) if !numbers.is_empty() => CoordinateChoice::List(numbers),
        _ => fail(format!(
            "invalid {flag} value: {raw} (expected {} or coordinate numbers from 1, separated by commas)",
            keyword.0
        )),
    }
}

/// The value after the flag `args[i]`, parsed as `T`.
fn flag_value<T: std::str::FromStr>(args: &[String], i: usize) -> T {
    let flag = &args[i];
    let Some(raw) = args.get(i + 1) else {
        fail(format!("{flag} requires a value"));
    };
    raw.parse()
        .unwrap_or_else(|_| fail(format!("invalid {flag} value: {raw}")))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        fail(format!(
            "Usage: {} <path_to_parm7> <path_to_nc> [--torsions-only] [--no-periodic] [--exact-constant] [--start N] [--stop N] [--stride N] [--mie-order 1|2|3|4] [--theiler-window W] [--tree-bonds] [--bonds heavy|tree|tree-unconstrained] [--tree mass|canonical] [--no-phase-angles] [--jacobian] [--estimator nn|qh|bqh|split] [--histogram-bins N] [--periodic-correlation complex|centered] [--per-coordinate] [--mutual-information] [--soft topology|detect|LIST] [--soft-estimator mie|mist|network|joint] [--network-parents N] [--hard-estimator qh|bqh|qh-mie1|qh-mie2] [--coupling gaussian|states|none] [--soft-features complex|cossin|harmonics] [--harmonics N] [--state-torsions auto|LIST] [--min-state-frames N]",
            args[0]
        ));
    }
    let top_path = Path::new(&args[1]);
    let traj_path = Path::new(&args[2]);

    let mut torsions_only = false;
    // None until a flag sets it: phase angles are on unless --legacy.
    let mut phase_angles: Option<bool> = None;
    let mut tree_bonds = false;
    let mut bond_set: Option<BondSet> = None;
    let mut tree_rule: Option<TreeRule> = None;
    let mut jacobian = false;
    let mut no_periodic = false;
    let mut exact_constant = false;
    let mut start: Option<usize> = None;
    let mut stop: Option<usize> = None;
    let mut legacy = false;
    let mut estimator = Estimator::NearestNeighbor;
    let mut histogram_bins: Option<usize> = None;
    let mut per_coordinate = false;
    let mut mutual_information = false;
    let mut periodic_correlation: Option<PeriodicCorrelation> = None;
    let mut bias_correction: Option<BiasCorrection> = None;
    // Split-estimator settings; None leaves the default.
    let mut soft: Option<CoordinateChoice> = None;
    let mut soft_estimator: Option<SoftEstimator> = None;
    let mut network_parents: Option<usize> = None;
    let mut hard_estimator: Option<HardEstimator> = None;
    let mut coupling: Option<CouplingFlag> = None;
    let mut soft_features: Option<SoftFeatures> = None;
    let mut harmonics: Option<usize> = None;
    let mut state_torsions: Option<CoordinateChoice> = None;
    let mut min_state_frames: Option<usize> = None;
    let mut mie_order = 2;
    let mut stride = 1usize;
    let mut theiler_window = 0usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--torsions-only" => {
                torsions_only = true;
                i += 1;
            }
            "--phase-angles" => {
                phase_angles = Some(true);
                i += 1;
            }
            "--no-phase-angles" => {
                phase_angles = Some(false);
                i += 1;
            }
            "--tree-bonds" => {
                tree_bonds = true;
                i += 1;
            }
            "--bonds" => {
                bond_set = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("heavy", BondSet::HeavyAtom),
                        ("tree", BondSet::Tree),
                        ("tree-unconstrained", BondSet::TreeUnconstrained),
                    ],
                ));
                i += 2;
            }
            "--tree" => {
                tree_rule = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("mass", TreeRule::MassOrdered),
                        ("canonical", TreeRule::Canonical),
                    ],
                ));
                i += 2;
            }
            "--jacobian" => {
                jacobian = true;
                i += 1;
            }
            "--no-periodic" => {
                no_periodic = true;
                i += 1;
            }
            "--exact-constant" => {
                exact_constant = true;
                i += 1;
            }
            "--start" => {
                start = Some(flag_value(&args, i));
                i += 2;
            }
            "--stop" => {
                stop = Some(flag_value(&args, i));
                i += 2;
            }
            "--mie-order" => {
                mie_order = flag_value(&args, i);
                i += 2;
            }
            "--stride" => {
                stride = flag_value(&args, i);
                i += 2;
            }
            "--theiler-window" => {
                theiler_window = flag_value(&args, i);
                i += 2;
            }
            "--legacy" => {
                legacy = true;
                i += 1;
            }
            "--estimator" => {
                if i + 1 >= args.len() {
                    fail("--estimator requires a value");
                }
                estimator = match args[i + 1].as_str() {
                    "nn" => Estimator::NearestNeighbor,
                    "qh" => Estimator::QuasiHarmonic,
                    "bqh" => Estimator::BoltzmannQuasiHarmonic,
                    "split" => Estimator::Split,
                    other => {
                        fail(format!(
                            "--estimator must be nn, qh, bqh or split, got {other}"
                        ));
                    }
                };
                i += 2;
            }
            "--mutual-information" => {
                mutual_information = true;
                i += 1;
            }
            "--per-coordinate" => {
                per_coordinate = true;
                i += 1;
            }
            "--soft" => {
                soft = Some(coordinate_choice(
                    &args,
                    i,
                    ("topology", CoordinateChoice::Topology),
                ));
                i += 2;
            }
            "--soft-estimator" => {
                soft_estimator = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("mie", SoftEstimator::NearestNeighborMie { order: 2 }),
                        ("mist", SoftEstimator::MutualInformationSpanningTree),
                        (
                            "network",
                            SoftEstimator::BayesianNetwork {
                                parents: DEFAULT_NETWORK_PARENTS,
                            },
                        ),
                        ("joint", SoftEstimator::JointNearestNeighbor),
                    ],
                ));
                i += 2;
            }
            "--hard-estimator" => {
                hard_estimator = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("qh", HardEstimator::QuasiHarmonic),
                        ("bqh", HardEstimator::BoltzmannQuasiHarmonic),
                        ("qh-mie1", HardEstimator::QuasiHarmonicMie1),
                        ("qh-mie2", HardEstimator::QuasiHarmonicMie2),
                    ],
                ));
                i += 2;
            }
            "--coupling" => {
                coupling = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("gaussian", CouplingFlag::Gaussian),
                        ("states", CouplingFlag::States),
                        ("none", CouplingFlag::None),
                    ],
                ));
                i += 2;
            }
            "--soft-features" => {
                soft_features = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("complex", SoftFeatures::Complex),
                        ("cossin", SoftFeatures::CosSin),
                        ("harmonics", SoftFeatures::default()),
                    ],
                ));
                i += 2;
            }
            "--harmonics" => {
                harmonics = Some(flag_value(&args, i));
                i += 2;
            }
            "--state-torsions" => {
                state_torsions = Some(coordinate_choice(
                    &args,
                    i,
                    ("auto", CoordinateChoice::Topology),
                ));
                i += 2;
            }
            "--min-state-frames" => {
                min_state_frames = Some(flag_value(&args, i));
                i += 2;
            }
            "--network-parents" => {
                network_parents = Some(flag_value(&args, i));
                i += 2;
            }
            "--bias" => {
                bias_correction = Some(flag_choice(
                    &args,
                    i,
                    &[
                        ("none", BiasCorrection::None),
                        ("wishart", BiasCorrection::Wishart),
                    ],
                ));
                i += 2;
            }
            "--periodic-correlation" => {
                if i + 1 >= args.len() {
                    fail("--periodic-correlation requires a value");
                }
                periodic_correlation = Some(match args[i + 1].as_str() {
                    "complex" => PeriodicCorrelation::Complex,
                    "centered" => PeriodicCorrelation::Centered,
                    other => {
                        fail(format!(
                            "--periodic-correlation must be complex or centered, got {other}"
                        ));
                    }
                });
                i += 2;
            }
            "--histogram-bins" => {
                histogram_bins = Some(flag_value(&args, i));
                i += 2;
            }
            other => {
                fail(format!("Unknown argument: {other}"));
            }
        }
    }
    if !(1..=4).contains(&mie_order) {
        fail("--mie-order must be 1, 2, 3, or 4");
    }
    if stride == 0 {
        fail("--stride must be at least 1");
    }
    if tree_bonds && bond_set.is_some() {
        fail("--tree-bonds is shorthand for --bonds tree; pass one or the other");
    }
    let bonds = bond_set.unwrap_or(if tree_bonds {
        BondSet::Tree
    } else {
        BondSet::HeavyAtom
    });
    if jacobian && (!bonds.is_tree() || torsions_only) {
        fail(
            "--jacobian requires --tree-bonds (or --bonds tree-unconstrained) and cannot be combined with --torsions-only",
        );
    }

    let split = estimator == Estimator::Split;
    let split_flags = [
        (soft.is_some(), "--soft"),
        (soft_estimator.is_some(), "--soft-estimator"),
        (hard_estimator.is_some(), "--hard-estimator"),
        (coupling.is_some(), "--coupling"),
        (soft_features.is_some(), "--soft-features"),
        (harmonics.is_some(), "--harmonics"),
        (state_torsions.is_some(), "--state-torsions"),
        (min_state_frames.is_some(), "--min-state-frames"),
    ];
    if !split && let Some((_, flag)) = split_flags.iter().find(|(given, _)| *given) {
        fail(format!("{flag} applies only to --estimator split"));
    }
    let mie_soft =
        soft_estimator.is_none_or(|e| matches!(e, SoftEstimator::NearestNeighborMie { .. }));
    if network_parents.is_some()
        && !(split && matches!(soft_estimator, Some(SoftEstimator::BayesianNetwork { .. })))
    {
        fail("--network-parents applies only to split with --soft-estimator network");
    }
    if mie_order != 2 && !(estimator == Estimator::NearestNeighbor || (split && mie_soft)) {
        fail("--mie-order applies only to --estimator nn, or to split with --soft-estimator mie");
    }
    let bqh_hard = hard_estimator == Some(HardEstimator::BoltzmannQuasiHarmonic);
    if histogram_bins.is_some()
        && !(estimator == Estimator::BoltzmannQuasiHarmonic || (split && bqh_hard))
    {
        fail(
            "--histogram-bins applies only to --estimator bqh, or to split with --hard-estimator bqh",
        );
    }
    if periodic_correlation.is_some() && estimator == Estimator::NearestNeighbor {
        fail("--periodic-correlation applies only to --estimator qh, bqh or split");
    }
    let coupling_flag = coupling.unwrap_or(CouplingFlag::Gaussian);
    if (soft_features.is_some() || harmonics.is_some()) && coupling_flag != CouplingFlag::Gaussian {
        fail("--soft-features and --harmonics apply only to --coupling gaussian");
    }
    if harmonics.is_some()
        && soft_features.is_some_and(|f| !matches!(f, SoftFeatures::Harmonics(_)))
    {
        fail("--harmonics applies only to --soft-features harmonics");
    }
    if harmonics == Some(0) {
        fail("--harmonics must be at least 1");
    }
    if (state_torsions.is_some() || min_state_frames.is_some())
        && coupling_flag != CouplingFlag::States
    {
        fail("--state-torsions and --min-state-frames apply only to --coupling states");
    }
    if split
        && soft
            .as_ref()
            .is_none_or(|soft| *soft == CoordinateChoice::Topology)
        && phase_angles == Some(false)
    {
        fail(
            "--soft topology needs phase angles: drop --no-phase-angles, or pass --soft detect or a list",
        );
    }
    if mutual_information && estimator != Estimator::NearestNeighbor {
        fail("--mutual-information applies only to --estimator nn");
    }
    if histogram_bins == Some(0) {
        fail("--histogram-bins must be at least 1");
    }
    if legacy {
        // The original Python estimator had none of these; each would make the
        // result differ from it.
        let conflicts = [
            (exact_constant, "--exact-constant"),
            (theiler_window != 0, "--theiler-window"),
            (mie_order != 2, "--mie-order other than 2"),
            (
                estimator != Estimator::NearestNeighbor,
                "--estimator other than nn",
            ),
            (tree_bonds, "--tree-bonds"),
            (bond_set.is_some(), "--bonds"),
            (tree_rule.is_some(), "--tree"),
            (phase_angles == Some(true), "--phase-angles"),
            (jacobian, "--jacobian"),
            (per_coordinate, "--per-coordinate"),
            (mutual_information, "--mutual-information"),
        ];
        for (given, flag) in conflicts {
            if given {
                fail(format!("{flag} cannot be combined with --legacy"));
            }
        }
    }

    let coordinate_options = CoordinateOptions {
        bonds,
        tree: tree_rule.unwrap_or_default(),
        // The original Python implementation used the raw torsions.
        phase_angles: phase_angles.unwrap_or(!legacy),
        geometry: if legacy {
            Geometry::Legacy
        } else {
            Geometry::Standard
        },
    };
    let mut internal = InternalCoordinates::with_options(top_path, coordinate_options)
        .unwrap_or_else(|err| {
            fail(format!(
                "Cannot read topology {}: {err}",
                top_path.display()
            ))
        });
    internal
        .calculate_internal_coords(traj_path, stop.unwrap_or(usize::MAX), torsions_only)
        .unwrap_or_else(|err| {
            fail(format!(
                "Cannot read trajectory {}: {err}",
                traj_path.display()
            ))
        });
    let start = start.unwrap_or(0);
    let one_d_data = internal
        .columns(start, stride)
        .unwrap_or_else(|err| fail(format!("Cannot select frames: {err}")));
    // Look these up once: the tables below describe every coordinate.
    let kinds = internal.coordinate_kinds();
    let atoms = internal.coordinate_atoms();
    let metrics = if no_periodic {
        vec![CoordinateMetric::Linear; internal.dim]
    } else {
        internal.coordinate_metrics()
    };
    let used_frames = one_d_data[0].len();
    let options = EntropyOptions {
        metrics: Some(&metrics),
        constant: if exact_constant {
            FiniteSampleConstant::Exact
        } else {
            FiniteSampleConstant::Asymptotic
        },
        theiler_window,
    };
    let quasi_harmonic = QuasiHarmonicOptions {
        bias: bias_correction.unwrap_or_default(),
        effective_frames: None,
        marginals: match histogram_bins {
            Some(bins) => MarginalEstimator::Histogram { bins },
            None => MarginalEstimator::NearestNeighbor,
        },
        periodic: periodic_correlation.unwrap_or_default(),
    };
    const FAILED: &str = "Entropy calculation failed";

    // The total, and for QH and BQH the first-order and correlation terms it
    // is the sum of.
    // With --per-coordinate, each coordinate's shares of the split entropy.
    let mut split_shares = None;
    let (entropy, terms) = match estimator {
        Estimator::Split => {
            let dim = internal.dim;
            let to_indices = |flag: &str, numbers: &[usize]| -> Vec<usize> {
                if let Some(&bad) = numbers.iter().find(|&&n| n > dim) {
                    fail(format!(
                        "{flag} coordinate {bad} is beyond the {dim} coordinates"
                    ));
                }
                numbers.iter().map(|n| n - 1).collect()
            };
            let topology_soft = internal.soft_torsions();
            let symmetric = internal.symmetric_rotor_torsions();
            let soft_list = match soft.clone().unwrap_or(CoordinateChoice::Topology) {
                CoordinateChoice::List(numbers) => Some(to_indices("--soft", &numbers)),
                _ => None,
            };
            let defining = match &state_torsions {
                Some(CoordinateChoice::List(numbers)) => {
                    Some(to_indices("--state-torsions", numbers))
                }
                _ => None,
            };
            let mut settings = SplitOptions::new(match (&soft, &soft_list) {
                (_, Some(list)) => SoftSelection::Indices(list),
                (Some(CoordinateChoice::Detect), _) => SoftSelection::Detect { candidates: None },
                _ => SoftSelection::Indices(&topology_soft),
            });
            if let Some(estimator) = soft_estimator {
                settings.soft_estimator = match estimator {
                    SoftEstimator::NearestNeighborMie { .. } => {
                        SoftEstimator::NearestNeighborMie { order: mie_order }
                    }
                    SoftEstimator::BayesianNetwork { .. } => SoftEstimator::BayesianNetwork {
                        parents: network_parents.unwrap_or(DEFAULT_NETWORK_PARENTS),
                    },
                    other => other,
                };
            } else {
                settings.soft_estimator = SoftEstimator::NearestNeighborMie { order: mie_order };
            }
            if let Some(estimator) = hard_estimator {
                settings.hard_estimator = estimator;
            }
            settings.hard.marginals = quasi_harmonic.marginals;
            settings.hard.bias = quasi_harmonic.bias;
            settings.hard.effective_frames = quasi_harmonic.effective_frames;
            if let Some(periodic) = periodic_correlation {
                settings.hard.periodic = periodic;
            }
            settings.coupling = match coupling_flag {
                CouplingFlag::Gaussian => Coupling::Gaussian {
                    features: match (soft_features, harmonics) {
                        (Some(SoftFeatures::Harmonics(_)) | None, Some(count)) => {
                            SoftFeatures::Harmonics(count)
                        }
                        (features, _) => features.unwrap_or_default(),
                    },
                },
                CouplingFlag::None => Coupling::None,
                CouplingFlag::States => Coupling::PerState(StateOptions {
                    definition: StateDefinition::Detected {
                        defining: defining.as_deref(),
                        exclude: if defining.is_some() { &[] } else { &symmetric },
                    },
                    min_frames: min_state_frames.unwrap_or(StateOptions::default().min_frames),
                    ..Default::default()
                }),
            };
            let result = if per_coordinate {
                let shares = or_fail(
                    estimate_coordinate_split_entropy(
                        &one_d_data,
                        used_frames,
                        &settings,
                        &options,
                    ),
                    FAILED,
                );
                let result = shares.split.clone();
                split_shares = Some(shares);
                result
            } else {
                or_fail(
                    estimate_split_entropy(&one_d_data, used_frames, &settings, &options),
                    FAILED,
                )
            };
            let numbers = |indices: &[usize]| {
                indices
                    .iter()
                    .map(|i| (i + 1).to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            println!("Soft coordinates = {}", numbers(&result.soft_coordinates));
            if result.soft_coordinates.len() > LARGE_SOFT_SET {
                eprintln!(
                    "Warning: {}",
                    large_soft_set_warning(result.soft_coordinates.len())
                );
            }
            if let Some(states) = &result.states {
                println!("state\tbasins\tframes\tpopulation\thard_entropy");
                for (index, state) in states.states.iter().enumerate() {
                    let basins = state
                        .basins
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    let entropy = state
                        .hard_entropy
                        .map_or("dropped".to_string(), |e| e.to_string());
                    println!(
                        "{}\t{basins}\t{}\t{}\t{entropy}",
                        index + 1,
                        state.frames,
                        state.population
                    );
                }
                println!("State-defining coordinates = {}", numbers(&states.defining));
                println!("Dropped frames = {}", states.dropped_frames);
                if states.dropped_population > 0.05 {
                    eprintln!(
                        "Warning: {:.1}% of the frames are in states too small to use; consider --min-state-frames or fewer --state-torsions",
                        100.0 * states.dropped_population
                    );
                }
            }
            println!("Soft entropy = {}", result.soft);
            println!("Hard entropy = {}", result.hard);
            println!("Coupling term = {}", result.coupling);
            (result.total, None)
        }
        _ if legacy => (
            or_fail(legacy_entropy(&one_d_data, used_frames), FAILED),
            None,
        ),
        Estimator::NearestNeighbor => (
            or_fail(
                calculate_entropy(&one_d_data, used_frames, mie_order, &options),
                FAILED,
            ),
            None,
        ),
        Estimator::QuasiHarmonic | Estimator::BoltzmannQuasiHarmonic => {
            let result = or_fail(
                estimate_quasi_harmonic_entropy(
                    &one_d_data,
                    used_frames,
                    &quasi_harmonic,
                    &options,
                ),
                FAILED,
            );
            if estimator == Estimator::QuasiHarmonic {
                (
                    result.quasi_harmonic,
                    Some((result.first_order_gaussian, result.correlation)),
                )
            } else {
                (
                    result.boltzmann_quasi_harmonic,
                    Some((result.first_order_boltzmann, result.correlation)),
                )
            }
        }
    };

    let describe = |index: usize| {
        let kind = kinds[index].name();
        // Atoms are numbered from 1, as in Amber and cpptraj.
        let atoms = atoms[index]
            .iter()
            .map(|atom| (atom + 1).to_string())
            .collect::<Vec<_>>()
            .join("-");
        format!("{}\t{kind}\t{atoms}", index + 1)
    };
    if let Some(shares) = &split_shares {
        println!("coordinate\ttype\tatoms\tentropy\tset\town\tcoupling");
        for index in 0..shares.total.len() {
            let set = if shares.split.soft_coordinates.contains(&index) {
                "soft"
            } else {
                "hard"
            };
            println!(
                "{}\t{}\t{set}\t{}\t{}",
                describe(index),
                shares.total[index],
                shares.own[index],
                shares.coupling[index]
            );
        }
    } else if per_coordinate {
        const PER_COORDINATE_FAILED: &str = "Per-coordinate entropy calculation failed";
        if estimator == Estimator::NearestNeighbor {
            let values = or_fail(
                estimate_coordinate_entropy(&one_d_data, used_frames, mie_order, &options),
                PER_COORDINATE_FAILED,
            );
            println!("coordinate\ttype\tatoms\tentropy");
            for (index, value) in values.iter().enumerate() {
                println!("{}\t{value}", describe(index));
            }
        } else {
            let shares = or_fail(
                estimate_coordinate_quasi_harmonic_entropy(
                    &one_d_data,
                    used_frames,
                    &quasi_harmonic,
                    &options,
                ),
                PER_COORDINATE_FAILED,
            );
            let (totals, first_order) = if estimator == Estimator::QuasiHarmonic {
                (&shares.quasi_harmonic, &shares.first_order_gaussian)
            } else {
                (
                    &shares.boltzmann_quasi_harmonic,
                    &shares.first_order_boltzmann,
                )
            };
            println!("coordinate\ttype\tatoms\tentropy\tfirst_order\tcorrelation");
            for index in 0..totals.len() {
                println!(
                    "{}\t{}\t{}\t{}",
                    describe(index),
                    totals[index],
                    first_order[index],
                    shares.correlation[index]
                );
            }
        }
    }
    if mutual_information {
        let values = or_fail(
            estimate_coordinate_mutual_information(&one_d_data, used_frames, &options),
            "Mutual information calculation failed",
        );
        println!("coordinate_a\tcoordinate_b\tmutual_information");
        let pairs = (0..internal.dim).flat_map(|i| ((i + 1)..internal.dim).map(move |j| (i, j)));
        for ((i, j), value) in pairs.zip(&values) {
            println!("{}\t{}\t{value}", i + 1, j + 1);
        }
    }
    if let Some((first_order, correlation)) = terms {
        println!("First-order entropy = {first_order}");
        println!("Correlation term = {correlation}");
    }
    if jacobian {
        // Over the same frames the entropy used.
        let log_jacobian = internal
            .log_jacobian()
            .expect("a tree bond set without --torsions-only was checked above");
        let mean =
            log_jacobian.iter().skip(start).step_by(stride).sum::<f64>() / used_frames as f64;
        println!("Mean ln J = {}", mean);
    }
    // Last, so scripts that read the final "= value" still get the entropy.
    println!("Total entropy = {}", entropy);
}
