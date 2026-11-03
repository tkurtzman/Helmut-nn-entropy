//! A coordinate set that can be written out and read back.
//!
//! The crate normally builds its own internal coordinates from a topology and
//! a Cartesian trajectory. That ties the definition of the coordinates to this
//! implementation, which is not always what you want: a collaborator may
//! already have a coordinate list from another program, a trajectory may be in
//! a format the reader here does not speak, or the coordinates may be the
//! thing under study and worth freezing in a file that outlives any one run.
//!
//! This module is the bridge. [`CoordinateSet`] holds exactly what the
//! estimators need about a set of coordinates — the atoms behind each one, its
//! kind, the torsion a phase angle is measured from, and which coordinates are
//! on rings, about symmetric rotors or in the default soft set — and can be
//! built either from an [`InternalCoordinates`] or from a pair of files.
//!
//! # The files
//!
//! The definitions are a tab-separated table with a `#` comment header, one
//! row per coordinate, coordinates and atoms numbered from 1 as in Amber and
//! cpptraj. The columns are separated by single tabs; they are drawn here
//! with `|` only so that the example lines up:
//!
//! ```text
//! # molentropy coordinate set 1
//! # bonds: tree-unconstrained
//! coordinate | kind    | atoms     | reference | ring | symmetric | soft
//! 1          | bond    | 2-5       | -         | 0    | 0         | 0
//! 30         | torsion | 5-7-9-15  | -         | 0    | 0         | 1
//! 31         | phase   | 11-9-7-5  | 30        | 0    | 0         | 0
//! ```
//!
//! The table is self-contained: nothing here needs the topology back. The
//! header lines carry provenance, and `bonds:` is read, because the BAT
//! Jacobian is defined only for a tree bond set.
//!
//! The values are one row per frame and one column per coordinate, either a
//! NumPy `.npy` array of `float64` or whitespace-separated text. Bond lengths
//! are in the trajectory's length unit, angles and torsions in radians, and
//! phase angles already wrapped to (-pi, pi].

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::CoordinateMetric;
use crate::bat_library::{BondSet, CoordinateKind, InternalCoordinates, Result};

/// The first line of a definition file, and the format it promises.
const MAGIC: &str = "# molentropy coordinate set 1";

/// The column header of a definition file.
const HEADER: &str = "coordinate\tkind\tatoms\treference\tring\tsymmetric\tsoft";

/// A set of internal coordinates and their values, however they were made.
///
/// Build it from the crate's own generator with
/// [`from_internal`](Self::from_internal), or from files written by any
/// program with [`read`](Self::read). Either way the accessors are the ones
/// the estimators take.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSet {
    /// Atom indices behind each coordinate, counted from 0: two for a bond,
    /// three for an angle, four for a torsion or phase angle.
    atoms: Vec<Vec<usize>>,
    kinds: Vec<CoordinateKind>,
    /// Whether each coordinate's central bond lies on a ring.
    ring: Vec<bool>,
    /// Whether each coordinate turns about a symmetric rotor such as a methyl.
    symmetric: Vec<bool>,
    /// Whether each coordinate is in the default soft set.
    soft: Vec<bool>,
    /// Which bonds the set was built from, which decides whether the Jacobian
    /// is defined.
    bonds: BondSet,
    /// Number of coordinates.
    pub dim: usize,
    /// Coordinate values, `values[frame][coordinate]`.
    pub values: Vec<Vec<f64>>,
}

impl CoordinateSet {
    /// The coordinates the crate's own generator built, with whatever frames
    /// have been read into it. Takes the generator by value because the values
    /// move across rather than being copied.
    pub fn from_internal(internal: InternalCoordinates) -> Self {
        let atoms = internal.coordinate_atoms();
        let kinds = internal.coordinate_kinds();
        let central = |index: usize| -> Option<(usize, usize)> {
            let entry = &atoms[index];
            (entry.len() == 4).then(|| (entry[1].min(entry[2]), entry[1].max(entry[2])))
        };
        let bonds_of = |indices: Vec<usize>| -> HashSet<(usize, usize)> {
            indices.into_iter().filter_map(central).collect()
        };
        // Flag every torsion about the bond, not only the primary one, so that
        // the flags describe the bond rather than this coordinate's place in
        // the group. The accessors below narrow them again.
        let ring_bonds = bonds_of(internal.ring_torsions());
        let symmetric_bonds = bonds_of(internal.symmetric_rotor_torsions());
        let soft: HashSet<usize> = internal.soft_torsions().into_iter().collect();

        let dim = atoms.len();
        let about = |index: usize, set: &HashSet<(usize, usize)>| {
            central(index).is_some_and(|bond| set.contains(&bond))
        };
        CoordinateSet {
            ring: (0..dim).map(|i| about(i, &ring_bonds)).collect(),
            symmetric: (0..dim).map(|i| about(i, &symmetric_bonds)).collect(),
            soft: (0..dim).map(|i| soft.contains(&i)).collect(),
            bonds: internal.options().bonds,
            atoms,
            kinds,
            dim,
            values: internal.int_coords,
        }
    }

    /// A coordinate set assembled by hand, for coordinates that came from
    /// somewhere other than a file or this crate's generator.
    ///
    /// `ring`, `symmetric` and `soft` describe the bond each torsion turns
    /// about and which of them the split estimator takes as soft; they may be
    /// empty, which means none. Fails if the pieces do not describe one
    /// consistent set.
    pub fn new(
        atoms: Vec<Vec<usize>>,
        kinds: Vec<CoordinateKind>,
        ring: Vec<bool>,
        symmetric: Vec<bool>,
        soft: Vec<bool>,
        bonds: BondSet,
        values: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let dim = atoms.len();
        if dim == 0 {
            return Err("a coordinate set needs at least one coordinate".into());
        }
        if kinds.len() != dim {
            return Err(format!("{dim} coordinates but {} kinds", kinds.len()).into());
        }
        let flags = |name: &str, given: Vec<bool>| -> Result<Vec<bool>> {
            match given.len() {
                0 => Ok(vec![false; dim]),
                n if n == dim => Ok(given),
                n => Err(format!("{dim} coordinates but {n} {name} flags").into()),
            }
        };
        for (index, (entry, kind)) in atoms.iter().zip(&kinds).enumerate() {
            let expected = match kind {
                CoordinateKind::Bond => 2,
                CoordinateKind::Angle => 3,
                _ => 4,
            };
            if entry.len() != expected {
                return Err(format!(
                    "coordinate {} is a {} and needs {expected} atoms, not {}",
                    index + 1,
                    kind.name(),
                    entry.len()
                )
                .into());
            }
            if let CoordinateKind::PhaseAngle { reference } = *kind {
                if reference >= index {
                    return Err(format!(
                        "coordinate {} is a phase angle of coordinate {}, which is not earlier",
                        index + 1,
                        reference + 1
                    )
                    .into());
                }
                if kinds[reference] != CoordinateKind::Torsion {
                    return Err(format!(
                        "coordinate {} is a phase angle of coordinate {}, which is not a torsion",
                        index + 1,
                        reference + 1
                    )
                    .into());
                }
            }
        }
        if let Some(frame) = values.iter().position(|frame| frame.len() != dim) {
            return Err(format!(
                "frame {} has {} values, but the set has {dim} coordinates",
                frame + 1,
                values[frame].len()
            )
            .into());
        }
        Ok(CoordinateSet {
            ring: flags("ring", ring)?,
            symmetric: flags("symmetric", symmetric)?,
            soft: flags("soft", soft)?,
            atoms,
            kinds,
            bonds,
            dim,
            values,
        })
    }

    /// The coordinate set defined by `definitions`, with the values in
    /// `values` when one is given.
    ///
    /// The values file is read as a NumPy array when its name ends in `.npy`
    /// and as whitespace-separated text otherwise. Fails if the two files
    /// disagree about the number of coordinates.
    pub fn read(definitions: &Path, values: Option<&Path>) -> Result<Self> {
        let mut set = Self::read_definitions(definitions)?;
        if let Some(path) = values {
            let rows = read_values(path)?;
            for (frame, row) in rows.iter().enumerate() {
                if row.len() != set.dim {
                    return Err(format!(
                        "{}: frame {} has {} values, but {} defines {} coordinates",
                        path.display(),
                        frame + 1,
                        row.len(),
                        definitions.display(),
                        set.dim
                    )
                    .into());
                }
            }
            set.values = rows;
        }
        Ok(set)
    }

    /// Parses a definition table, ignoring `#` comments and the column header.
    fn read_definitions(path: &Path) -> Result<Self> {
        let file =
            File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
        let mut bonds = BondSet::HeavyAtom;
        let (mut atoms, mut kinds) = (Vec::new(), Vec::new());
        let (mut ring, mut symmetric, mut soft) = (Vec::new(), Vec::new(), Vec::new());
        // Phase references name a coordinate, resolved once every row is in.
        let mut references: Vec<Option<usize>> = Vec::new();
        for (number, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|err| format!("{}: {err}", path.display()))?;
            let line = line.trim_end();
            let at = |what: String| format!("{}:{}: {what}", path.display(), number + 1);
            if let Some(rest) = line.strip_prefix('#') {
                if let Some(value) = rest.trim().strip_prefix("bonds:") {
                    bonds = match value.trim() {
                        "heavy" => BondSet::HeavyAtom,
                        "tree" => BondSet::Tree,
                        "tree-unconstrained" => BondSet::TreeUnconstrained,
                        other => return Err(at(format!("unknown bond set {other}")).into()),
                    };
                }
                continue;
            }
            if line.is_empty() || line.starts_with("coordinate\t") {
                continue;
            }
            let cells: Vec<&str> = line.split('\t').collect();
            if cells.len() != 7 {
                return Err(at(format!(
                    "expected 7 tab-separated columns, found {}",
                    cells.len()
                ))
                .into());
            }
            let index: usize = cells[0]
                .trim()
                .parse()
                .map_err(|_| at(format!("coordinate number {} is not a number", cells[0])))?;
            if index != atoms.len() + 1 {
                return Err(at(format!(
                    "coordinates must be numbered from 1 in order; expected {}, found {index}",
                    atoms.len() + 1
                ))
                .into());
            }
            let entry = parse_atoms(cells[2]).map_err(&at)?;
            let kind = cells[1].trim();
            let expected = match kind {
                "bond" => 2,
                "angle" => 3,
                "torsion" | "phase" | "phase angle" => 4,
                other => return Err(at(format!("unknown coordinate kind {other}")).into()),
            };
            if entry.len() != expected {
                return Err(at(format!(
                    "a {kind} needs {expected} atoms, found {}",
                    entry.len()
                ))
                .into());
            }
            let reference = match cells[3].trim() {
                "-" | "" => None,
                raw => Some(
                    raw.parse::<usize>()
                        .map_err(|_| at(format!("reference {raw} is not a coordinate number")))?,
                ),
            };
            let phase = matches!(kind, "phase" | "phase angle");
            match (phase, reference) {
                (true, None) => {
                    return Err(at("a phase angle needs a reference torsion".into()).into());
                }
                (false, Some(_)) => {
                    return Err(at(format!("a {kind} cannot have a reference torsion")).into());
                }
                _ => {}
            }
            if let Some(reference) = reference
                && (reference == 0 || reference >= index)
            {
                return Err(at(format!(
                    "coordinate {index} references {reference}, which is not an earlier coordinate"
                ))
                .into());
            }
            references.push(reference);
            kinds.push(if phase {
                // Placeholder; the reference is rewritten to a 0-based index
                // once the whole table is known to be consistent.
                CoordinateKind::PhaseAngle { reference: 0 }
            } else {
                match kind {
                    "bond" => CoordinateKind::Bond,
                    "angle" => CoordinateKind::Angle,
                    _ => CoordinateKind::Torsion,
                }
            });
            atoms.push(entry);
            ring.push(parse_flag(cells[4]).map_err(&at)?);
            symmetric.push(parse_flag(cells[5]).map_err(&at)?);
            soft.push(parse_flag(cells[6]).map_err(&at)?);
        }
        if atoms.is_empty() {
            return Err(format!("{}: no coordinates", path.display()).into());
        }
        for (index, reference) in references.iter().enumerate() {
            let Some(reference) = *reference else {
                continue;
            };
            let reference = reference - 1;
            if kinds[reference] != CoordinateKind::Torsion {
                return Err(format!(
                    "{}: coordinate {} is a phase angle of coordinate {}, which is not a torsion",
                    path.display(),
                    index + 1,
                    reference + 1
                )
                .into());
            }
            kinds[index] = CoordinateKind::PhaseAngle { reference };
        }
        let dim = atoms.len();
        Ok(CoordinateSet {
            atoms,
            kinds,
            ring,
            symmetric,
            soft,
            bonds,
            dim,
            values: Vec::new(),
        })
    }

    /// Writes the definition table to `path`, with `meta` as extra header
    /// comments in the order given.
    pub fn write_definitions(&self, path: &Path, meta: &[(&str, String)]) -> Result<()> {
        let file =
            File::create(path).map_err(|err| format!("cannot write {}: {err}", path.display()))?;
        let mut out = BufWriter::new(file);
        writeln!(out, "{MAGIC}")?;
        writeln!(
            out,
            "# bonds: {}",
            match self.bonds {
                BondSet::HeavyAtom => "heavy",
                BondSet::Tree => "tree",
                BondSet::TreeUnconstrained => "tree-unconstrained",
            }
        )?;
        for (key, value) in meta {
            writeln!(out, "# {key}: {value}")?;
        }
        writeln!(out, "{HEADER}")?;
        for index in 0..self.dim {
            let atoms = self.atoms[index]
                .iter()
                .map(|atom| (atom + 1).to_string())
                .collect::<Vec<_>>()
                .join("-");
            let (kind, reference) = match self.kinds[index] {
                CoordinateKind::Bond => ("bond", "-".to_string()),
                CoordinateKind::Angle => ("angle", "-".to_string()),
                CoordinateKind::Torsion => ("torsion", "-".to_string()),
                CoordinateKind::PhaseAngle { reference } => ("phase", (reference + 1).to_string()),
            };
            writeln!(
                out,
                "{}\t{kind}\t{atoms}\t{reference}\t{}\t{}\t{}",
                index + 1,
                self.ring[index] as u8,
                self.symmetric[index] as u8,
                self.soft[index] as u8
            )?;
        }
        out.flush()?;
        Ok(())
    }

    /// Writes the values to `path`, as a NumPy array when the name ends in
    /// `.npy` and as tab-separated text otherwise.
    pub fn write_values(&self, path: &Path) -> Result<()> {
        if self.values.is_empty() {
            return Err("there are no coordinate values to write".into());
        }
        if path.extension().is_some_and(|e| e == "npy") {
            write_npy(path, &self.values)
        } else {
            write_tsv(path, &self.values)
        }
    }

    /// The values of every coordinate on frames `start`, `start + stride`,
    /// ... of those read, as `columns[coordinate][frame]`: the layout the
    /// estimators take.
    pub fn columns(&self, start: usize, stride: usize) -> Result<Vec<Vec<f64>>> {
        if stride == 0 {
            return Err("stride must be at least 1".into());
        }
        if self.dim == 0 {
            return Err("no internal coordinates were generated".into());
        }
        let frames = self.values.len();
        if frames == 0 {
            return Err("no frames were read from the trajectory".into());
        }
        if start >= frames {
            return Err(format!("start {start} is beyond the {frames} frames read").into());
        }
        let selected: Vec<&Vec<f64>> = self.values.iter().skip(start).step_by(stride).collect();
        Ok((0..self.dim)
            .map(|coordinate| selected.iter().map(|frame| frame[coordinate]).collect())
            .collect())
    }

    /// The kind of each coordinate, with the reference torsion of each phase
    /// angle.
    pub fn coordinate_kinds(&self) -> Vec<CoordinateKind> {
        self.kinds.clone()
    }

    /// Atom indices of each coordinate, counted from 0.
    pub fn coordinate_atoms(&self) -> Vec<Vec<usize>> {
        self.atoms.clone()
    }

    /// The metric of each coordinate: periodic with period 2 pi for torsions
    /// and phase angles, linear for bonds and angles.
    pub fn coordinate_metrics(&self) -> Vec<CoordinateMetric> {
        self.kinds
            .iter()
            .map(|kind| match kind {
                CoordinateKind::Bond | CoordinateKind::Angle => CoordinateMetric::Linear,
                _ => CoordinateMetric::Periodic {
                    period: std::f64::consts::TAU,
                },
            })
            .collect()
    }

    /// The default soft set: the primary torsion about every non-ring
    /// rotatable bond, as the definition file flags them.
    pub fn soft_torsions(&self) -> Vec<usize> {
        (0..self.dim).filter(|&i| self.soft[i]).collect()
    }

    /// The soft torsions about symmetric rotors such as methyl groups: their
    /// basins are equivalent, so they define no conformational states.
    pub fn symmetric_rotor_torsions(&self) -> Vec<usize> {
        (0..self.dim)
            .filter(|&i| self.soft[i] && self.symmetric[i])
            .collect()
    }

    /// Torsions about bonds that lie on a ring.
    pub fn ring_torsions(&self) -> Vec<usize> {
        (0..self.dim)
            .filter(|&i| self.ring[i] && self.atoms[i].len() == 4)
            .collect()
    }

    /// ln J for each frame, as [`InternalCoordinates::log_jacobian`] defines
    /// it. Available only for a set built on a tree bond set, which the
    /// definition file's `bonds:` header records.
    pub fn log_jacobian(&self) -> Result<Vec<f64>> {
        if !self.bonds.is_tree() {
            return Err("the BAT Jacobian needs a tree bond set".into());
        }
        if !self
            .kinds
            .iter()
            .any(|kind| matches!(kind, CoordinateKind::Bond | CoordinateKind::Angle))
        {
            return Err("the BAT Jacobian needs bonds and angles, not torsions only".into());
        }
        Ok(self
            .values
            .iter()
            .map(|frame| {
                self.kinds
                    .iter()
                    .zip(frame)
                    .map(|(kind, value)| match kind {
                        CoordinateKind::Bond => 2.0 * value.ln(),
                        CoordinateKind::Angle => value.sin().ln(),
                        _ => 0.0,
                    })
                    .sum()
            })
            .collect())
    }
}

/// `1-2-3` as zero-based atom indices.
fn parse_atoms(cell: &str) -> std::result::Result<Vec<usize>, String> {
    cell.trim()
        .split('-')
        .map(|atom| match atom.trim().parse::<usize>() {
            Ok(0) => Err("atoms are numbered from 1".to_string()),
            Ok(number) => Ok(number - 1),
            Err(_) => Err(format!("atom {atom} is not a number")),
        })
        .collect()
}

fn parse_flag(cell: &str) -> std::result::Result<bool, String> {
    match cell.trim() {
        "0" => Ok(false),
        "1" => Ok(true),
        other => Err(format!("flag {other} must be 0 or 1")),
    }
}

// ---------------------------------------------------------------------------
// Value files
// ---------------------------------------------------------------------------

/// Reads a values file, as a NumPy array when the name ends in `.npy` and as
/// whitespace-separated text otherwise.
fn read_values(path: &Path) -> Result<Vec<Vec<f64>>> {
    let rows = if path.extension().is_some_and(|e| e == "npy") {
        read_npy(path)?
    } else {
        read_tsv(path)?
    };
    if rows.is_empty() {
        return Err(format!("{}: no frames", path.display()).into());
    }
    Ok(rows)
}

fn read_tsv(path: &Path) -> Result<Vec<Vec<f64>>> {
    let file = File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| format!("{}: {err}", path.display()))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: std::result::Result<Vec<f64>, _> =
            line.split_whitespace().map(str::parse::<f64>).collect();
        rows.push(
            row.map_err(|err| format!("{}:{}: not a number: {err}", path.display(), number + 1))?,
        );
    }
    Ok(rows)
}

fn write_tsv(path: &Path, values: &[Vec<f64>]) -> Result<()> {
    let file =
        File::create(path).map_err(|err| format!("cannot write {}: {err}", path.display()))?;
    let mut out = BufWriter::new(file);
    for frame in values {
        let mut first = true;
        for value in frame {
            if !first {
                out.write_all(b"\t")?;
            }
            first = false;
            // Enough digits to read back the same f64.
            write!(out, "{value:.17e}")?;
        }
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

/// The NumPy `.npy` header this crate writes and the shape it describes.
fn npy_header(rows: usize, columns: usize) -> Vec<u8> {
    let dict =
        format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {columns}), }}");
    // The data must start on a 64-byte boundary: 6 magic bytes, 2 version
    // bytes, 2 length bytes, then the padded dictionary ending in a newline.
    let mut header = dict.into_bytes();
    let prefix = 6 + 2 + 2;
    let padded = (prefix + header.len() + 1).div_ceil(64) * 64;
    header.resize(padded - prefix - 1, b' ');
    header.push(b'\n');
    header
}

fn write_npy(path: &Path, values: &[Vec<f64>]) -> Result<()> {
    let columns = values[0].len();
    if values.iter().any(|frame| frame.len() != columns) {
        return Err("every frame must have the same number of values".into());
    }
    let file =
        File::create(path).map_err(|err| format!("cannot write {}: {err}", path.display()))?;
    let mut out = BufWriter::new(file);
    let header = npy_header(values.len(), columns);
    out.write_all(b"\x93NUMPY\x01\x00")?;
    out.write_all(&(header.len() as u16).to_le_bytes())?;
    out.write_all(&header)?;
    for frame in values {
        for value in frame {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Reads a two-dimensional C-order NumPy array of little-endian `f8` or `f4`.
fn read_npy(path: &Path) -> Result<Vec<Vec<f64>>> {
    let mut file =
        File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .map_err(|err| format!("{}: {err}", path.display()))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a NumPy .npy file", path.display()).into());
    }
    let length = match magic[6] {
        1 => {
            let mut raw = [0u8; 2];
            file.read_exact(&mut raw)?;
            u16::from_le_bytes(raw) as usize
        }
        2 | 3 => {
            let mut raw = [0u8; 4];
            file.read_exact(&mut raw)?;
            u32::from_le_bytes(raw) as usize
        }
        other => {
            return Err(format!("{}: unsupported .npy version {other}", path.display()).into());
        }
    };
    let mut header = vec![0u8; length];
    file.read_exact(&mut header)?;
    let header = String::from_utf8(header)
        .map_err(|_| format!("{}: the .npy header is not text", path.display()))?;

    if header.contains("'fortran_order': True") {
        return Err(format!(
            "{}: the array is Fortran-ordered; save it with C order",
            path.display()
        )
        .into());
    }
    let width = if header.contains("'<f8'") || header.contains("'|f8'") {
        8
    } else if header.contains("'<f4'") || header.contains("'|f4'") {
        4
    } else {
        return Err(format!(
            "{}: the array must be float32 or float64 in little-endian order",
            path.display()
        )
        .into());
    };
    let shape = header
        .split_once("'shape':")
        .and_then(|(_, rest)| rest.split_once('('))
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(inside, _)| inside.to_string())
        .ok_or_else(|| format!("{}: the .npy header has no shape", path.display()))?;
    let dimensions: std::result::Result<Vec<usize>, _> = shape
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::parse::<usize>)
        .collect();
    let dimensions =
        dimensions.map_err(|_| format!("{}: the .npy shape is not numeric", path.display()))?;
    let [rows, columns] = dimensions[..] else {
        return Err(format!(
            "{}: expected a two-dimensional array, found {} dimensions",
            path.display(),
            dimensions.len()
        )
        .into());
    };

    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    if data.len() != rows * columns * width {
        return Err(format!(
            "{}: the array holds {} bytes, but its shape needs {}",
            path.display(),
            data.len(),
            rows * columns * width
        )
        .into());
    }
    let value = |chunk: &[u8]| -> f64 {
        if width == 8 {
            f64::from_le_bytes(chunk.try_into().expect("8 bytes"))
        } else {
            f32::from_le_bytes(chunk.try_into().expect("4 bytes")) as f64
        }
    };
    Ok(data
        .chunks_exact(columns * width)
        .map(|row| row.chunks_exact(width).map(value).collect())
        .collect())
}
