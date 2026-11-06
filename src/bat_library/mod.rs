//! Amber topologies and trajectories to bond-angle-torsion (BAT) coordinates.
//!
//! [`InternalCoordinates`] reads a `.parm7` topology, selects the solute
//! (water and massless virtual sites are excluded; molecules may appear in any
//! order), builds a BAT tree for each molecule, and evaluates its coordinates
//! on every frame of a NetCDF trajectory. [`CoordinateOptions`] chooses the bond
//! set, phase angles and the geometry formulas.
//!
//! The coordinates of each molecule, in order, are its bonds, its angles and
//! its torsions. Bonds and angles are linear coordinates, torsions periodic with
//! period 2 pi; [`InternalCoordinates::coordinate_metrics`] reports which.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::CoordinateMetric;

pub mod geometry;

use geometry::{internal_coordinates, internal_coordinates_f32};

/// Result of reading files and building coordinates.
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct Prmtop {
    natom: usize,
    masses: Vec<f64>,
    atom_types: Vec<String>,
    /// Per-atom PDB-style names (`N`, `CA`, `CB`, `HB1`, ...); empty when the
    /// topology has no ATOM_NAME section. Used only by [`TreeRule::Canonical`].
    atom_names: Vec<String>,
    /// Per-atom atomic numbers; empty when the topology predates ATOMIC_NUMBER.
    atomic_numbers: Vec<i64>,
    /// Per-atom residue labels; empty when RESIDUE_LABEL/POINTER are absent.
    residue_labels: Vec<String>,
    bonds: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct FormatSpec {
    count: usize,
    width: usize,
    kind: char,
}

fn parse_format_spec(line: &str) -> Result<FormatSpec> {
    let start = line.find('(').ok_or("missing '(' in %FORMAT")? + 1;
    let end = line.find(')').ok_or("missing ')' in %FORMAT")?;
    let inner = line[start..end].trim();
    let mut digits = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else {
            let count = digits.parse::<usize>()?;
            let kind = c;
            let mut width_digits = String::new();
            for c2 in chars {
                if c2.is_ascii_digit() {
                    width_digits.push(c2);
                } else {
                    break;
                }
            }
            let width = width_digits.parse::<usize>()?;
            return Ok(FormatSpec { count, width, kind });
        }
    }
    Err("invalid %FORMAT line".into())
}

fn parse_fixed_width_tokens(line: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let end = (i + width).min(bytes.len());
        let chunk = &bytes[i..end];
        let token = String::from_utf8_lossy(chunk).trim().to_string();
        if !token.is_empty() {
            tokens.push(token);
        }
        i += width;
    }
    tokens
}

fn parse_prmtop(path: &Path) -> Result<Prmtop> {
    let text = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().collect();
    let mut sections: HashMap<String, (FormatSpec, Vec<String>)> = HashMap::new();

    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("%FLAG") {
            let flag = line
                .split_whitespace()
                .nth(1)
                .ok_or("missing flag name")?
                .to_string();
            i += 1;
            if i >= lines.len() {
                return Err("unexpected end after %FLAG".into());
            }
            let fmt_line = lines[i].trim();
            if !fmt_line.starts_with("%FORMAT") {
                return Err("expected %FORMAT after %FLAG".into());
            }
            let fmt = parse_format_spec(fmt_line)?;
            i += 1;
            let mut tokens = Vec::new();
            while i < lines.len() && !lines[i].trim().starts_with("%FLAG") {
                tokens.extend(parse_fixed_width_tokens(lines[i], fmt.width));
                i += 1;
            }
            sections.insert(flag, (fmt, tokens));
        } else {
            i += 1;
        }
    }

    let pointers = sections
        .get("POINTERS")
        .ok_or("missing POINTERS section")?
        .1
        .iter()
        .map(|s| s.parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if pointers.len() < 32 {
        let mut padded = pointers;
        padded.resize(32, 0);
        return parse_prmtop_from_pointers(padded, &sections);
    }
    parse_prmtop_from_pointers(pointers, &sections)
}

fn parse_prmtop_from_pointers(
    pointers: Vec<i64>,
    sections: &HashMap<String, (FormatSpec, Vec<String>)>,
) -> Result<Prmtop> {
    let natom = pointers[0] as usize;
    let nbonh = pointers[2] as usize;
    let mbona = pointers[3] as usize;
    let nbona = pointers[12] as usize;

    let masses = sections
        .get("MASS")
        .ok_or("missing MASS section")?
        .1
        .iter()
        .take(natom)
        .map(|s| s.parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let atom_types = if let Some(sec) = sections.get("AMBER_ATOM_TYPE") {
        sec.1
            .iter()
            .take(natom)
            .map(|s| s.trim().to_string())
            .collect()
    } else if let Some(sec) = sections.get("ATOM_NAME") {
        sec.1
            .iter()
            .take(natom)
            .map(|s| s.trim().to_string())
            .collect()
    } else {
        return Err("missing AMBER_ATOM_TYPE and ATOM_NAME sections".into());
    };

    let atom_names = match sections.get("ATOM_NAME") {
        Some(sec) => sec
            .1
            .iter()
            .take(natom)
            .map(|s| s.trim().to_string())
            .collect(),
        None => Vec::new(),
    };

    let bond_h_tokens = sections
        .get("BONDS_INC_HYDROGEN")
        .map(|s| s.1.clone())
        .unwrap_or_default();
    let bond_no_h_tokens = sections
        .get("BONDS_WITHOUT_HYDROGEN")
        .map(|s| s.1.clone())
        .unwrap_or_default();

    let bond_h_count = if bond_h_tokens.len() >= nbonh * 3 {
        nbonh
    } else {
        bond_h_tokens.len() / 3
    };
    let bond_no_h_count = if bond_no_h_tokens.len() >= nbona * 3 {
        nbona
    } else if bond_no_h_tokens.len() >= mbona * 3 {
        mbona
    } else {
        bond_no_h_tokens.len() / 3
    };

    let atomic_numbers = match sections.get("ATOMIC_NUMBER") {
        Some(sec) => sec
            .1
            .iter()
            .take(natom)
            .map(|s| s.parse::<i64>())
            .collect::<std::result::Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };

    let residue_labels = match (
        sections.get("RESIDUE_LABEL"),
        sections.get("RESIDUE_POINTER"),
    ) {
        (Some(labels), Some(pointers)) => {
            let starts = pointers
                .1
                .iter()
                .map(|s| s.parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if starts.len() != labels.1.len() {
                return Err("RESIDUE_LABEL and RESIDUE_POINTER lengths differ".into());
            }
            let mut per_atom = vec![String::new(); natom];
            for (residue, &start) in starts.iter().enumerate() {
                // RESIDUE_POINTER is one-based; a residue runs to the next start.
                let first = start
                    .checked_sub(1)
                    .ok_or("RESIDUE_POINTER entry of zero")?;
                let end = starts.get(residue + 1).map_or(natom, |&next| next - 1);
                if first > end || end > natom {
                    return Err("RESIDUE_POINTER is out of order or out of range".into());
                }
                for label in &mut per_atom[first..end] {
                    label.clone_from(&labels.1[residue]);
                }
            }
            per_atom
        }
        _ => Vec::new(),
    };

    let mut bonds = Vec::new();
    for i in 0..bond_h_count {
        let a = bond_h_tokens[i * 3].parse::<i64>()?;
        let b = bond_h_tokens[i * 3 + 1].parse::<i64>()?;
        let ai = (a.unsigned_abs() as usize) / 3;
        let bi = (b.unsigned_abs() as usize) / 3;
        bonds.push((ai, bi));
    }
    for i in 0..bond_no_h_count {
        let a = bond_no_h_tokens[i * 3].parse::<i64>()?;
        let b = bond_no_h_tokens[i * 3 + 1].parse::<i64>()?;
        let ai = (a.unsigned_abs() as usize) / 3;
        let bi = (b.unsigned_abs() as usize) / 3;
        bonds.push((ai, bi));
    }

    Ok(Prmtop {
        natom,
        masses,
        atom_types,
        atom_names,
        atomic_numbers,
        residue_labels,
        bonds,
    })
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NetcdfDim {
    name: String,
    len: u64,
}

#[derive(Debug, Clone)]
struct NetcdfVar {
    name: String,
    dim_ids: Vec<usize>,
    vartype: u32,
    vsize: u64,
    begin: u64,
    is_record: bool,
}

#[allow(dead_code)]
struct NetcdfReader {
    file: File,
    /// Atom count the topology declares; the trajectory must match it.
    expected_atoms: usize,
    version: u8,
    numrecs: u64,
    dims: Vec<NetcdfDim>,
    vars: Vec<NetcdfVar>,
    record_size: u64,
}

/// NetCDF `numrecs` value for a file still being written.
const STREAMING_NUMRECS: u32 = u32::MAX;

impl NetcdfReader {
    fn open(path: &Path, expected_atoms: usize) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic[0..3] != b"CDF" {
            return Err("not a NetCDF classic file".into());
        }
        let version = magic[3];
        if version != 1 && version != 2 {
            return Err("unsupported NetCDF version".into());
        }
        let declared_records = Self::read_u32(&mut file)?;

        let dims = Self::read_dim_list(&mut file)?;
        Self::skip_attr_list(&mut file)?;
        let vars = Self::read_var_list(&mut file, version, &dims)?;
        let record_size: u64 = vars.iter().filter(|v| v.is_record).map(|v| v.vsize).sum();

        // 0xFFFFFFFF marks a streaming (unfinalized) file whose record count
        // was never written back; count the complete records actually present.
        let numrecs = if declared_records == STREAMING_NUMRECS {
            let first_record = vars
                .iter()
                .filter(|v| v.is_record)
                .map(|v| v.begin)
                .min()
                .ok_or("streaming NetCDF file has no record variables")?;
            if record_size == 0 {
                return Err("streaming NetCDF file has zero-sized records".into());
            }
            let file_len = file.metadata()?.len();
            file_len.saturating_sub(first_record) / record_size
        } else {
            declared_records as u64
        };

        Ok(NetcdfReader {
            file,
            expected_atoms,
            version,
            numrecs,
            dims,
            vars,
            record_size,
        })
    }

    fn coordinates_var(&self) -> Result<NetcdfVar> {
        self.vars
            .iter()
            .find(|v| v.name == "coordinates")
            .cloned()
            .ok_or("coordinates variable not found".into())
    }

    fn coordinates_vartype(&self) -> Result<u32> {
        Ok(self.coordinates_var()?.vartype)
    }

    /// Read `atoms` (sorted, zero-based indices into the file's atom
    /// dimension) from the first `frames` frames, decoding each value with
    /// `decode`. Only the span from the first to the last requested atom is
    /// read, so a solute stored ahead of its solvent costs no more than before.
    fn read_coordinates<T: Copy + Default>(
        &mut self,
        frames: usize,
        atoms: &[usize],
        bytes_per_value: usize,
        decode: impl Fn(&[u8]) -> T,
    ) -> Result<Vec<Vec<[T; 3]>>> {
        let var = self.coordinates_var()?;
        let dims: Vec<u64> = var.dim_ids.iter().map(|&i| self.dims[i].len).collect();

        let (frame_count, atom_dim, spatial_dim) = if var.is_record {
            if dims.len() != 3 {
                return Err("coordinates variable must be (frame, atom, spatial)".into());
            }
            (frames.min(self.numrecs as usize), dims[1], dims[2])
        } else {
            if dims.len() != 2 {
                return Err("coordinates variable must be (atom, spatial)".into());
            }
            (1usize, dims[0], dims[1])
        };
        if spatial_dim != 3 {
            return Err("coordinates spatial dimension is not 3".into());
        }
        if atom_dim != self.expected_atoms as u64 {
            return Err(format!(
                "trajectory has {atom_dim} atoms but the topology has {}",
                self.expected_atoms
            )
            .into());
        }
        let (Some(&first), Some(&last)) = (atoms.first(), atoms.last()) else {
            return Ok(vec![Vec::new(); frame_count]);
        };
        debug_assert!(atoms.windows(2).all(|w| w[0] < w[1]));
        if last as u64 >= atom_dim {
            return Err(
                format!("topology selects atom {last} but the trajectory has {atom_dim}").into(),
            );
        }

        let atom_bytes = 3 * bytes_per_value;
        let mut span = vec![0u8; (last - first + 1) * atom_bytes];
        let mut frames_out = Vec::with_capacity(frame_count);
        for frame_idx in 0..frame_count {
            let frame_start = if var.is_record {
                var.begin + (frame_idx as u64) * self.record_size
            } else {
                var.begin
            };
            self.file
                .seek(SeekFrom::Start(frame_start + (first * atom_bytes) as u64))?;
            self.file.read_exact(&mut span)?;

            let frame = atoms
                .iter()
                .map(|&atom| {
                    let at = (atom - first) * atom_bytes;
                    let mut xyz = [T::default(); 3];
                    for (axis, value) in xyz.iter_mut().enumerate() {
                        let offset = at + axis * bytes_per_value;
                        *value = decode(&span[offset..offset + bytes_per_value]);
                    }
                    xyz
                })
                .collect();
            frames_out.push(frame);
        }
        Ok(frames_out)
    }

    fn read_coordinates_f64(
        &mut self,
        frames: usize,
        atoms: &[usize],
    ) -> Result<Vec<Vec<[f64; 3]>>> {
        match self.coordinates_vartype()? {
            5 => self.read_coordinates(frames, atoms, 4, |b| {
                f32::from_be_bytes(b.try_into().unwrap()) as f64
            }),
            6 => self.read_coordinates(frames, atoms, 8, |b| {
                f64::from_be_bytes(b.try_into().unwrap())
            }),
            _ => Err("unsupported coordinates data type".into()),
        }
    }

    fn read_coordinates_f32(
        &mut self,
        frames: usize,
        atoms: &[usize],
    ) -> Result<Vec<Vec<[f32; 3]>>> {
        if self.coordinates_vartype()? != 5 {
            return Err("coordinates variable is not float".into());
        }
        self.read_coordinates(frames, atoms, 4, |b| {
            f32::from_be_bytes(b.try_into().unwrap())
        })
    }

    fn read_dim_list(file: &mut File) -> Result<Vec<NetcdfDim>> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
            Self::read_absent_count(file)?;
            return Ok(Vec::new());
        }
        if tag != 10 {
            return Err("unexpected tag in dimension list".into());
        }
        let count = Self::read_u32(file)? as usize;
        let mut dims = Vec::with_capacity(count);
        for _ in 0..count {
            let name = Self::read_string(file)?;
            let len = Self::read_u32(file)? as u64;
            dims.push(NetcdfDim { name, len });
        }
        Ok(dims)
    }

    fn skip_attr_list(file: &mut File) -> Result<()> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
            Self::read_absent_count(file)?;
            return Ok(());
        }
        if tag != 12 {
            return Err("unexpected tag in attribute list".into());
        }
        let count = Self::read_u32(file)? as usize;
        for _ in 0..count {
            let _name = Self::read_string(file)?;
            let typ = Self::read_u32(file)?;
            let len = Self::read_u32(file)? as u64;
            let bytes = Self::type_size(typ)? as u64 * len;
            let padded = Self::pad4(bytes);
            file.seek(SeekFrom::Current(padded as i64))?;
        }
        Ok(())
    }

    fn read_var_list(file: &mut File, version: u8, dims: &[NetcdfDim]) -> Result<Vec<NetcdfVar>> {
        let tag = Self::read_u32(file)?;
        if tag == 0 {
            Self::read_absent_count(file)?;
            return Ok(Vec::new());
        }
        if tag != 11 {
            return Err("unexpected tag in variable list".into());
        }
        let count = Self::read_u32(file)? as usize;
        let mut vars = Vec::with_capacity(count);
        for _ in 0..count {
            let name = Self::read_string(file)?;
            let dim_count = Self::read_u32(file)? as usize;
            let mut dim_ids = Vec::with_capacity(dim_count);
            for _ in 0..dim_count {
                dim_ids.push(Self::read_u32(file)? as usize);
            }
            Self::skip_attr_list(file)?;
            let vartype = Self::read_u32(file)?;
            if !(1..=6).contains(&vartype) {
                return Err("unexpected NetCDF variable type".into());
            }
            let vsize = Self::read_u32(file)? as u64;
            let begin = if version == 1 {
                Self::read_u32(file)? as u64
            } else {
                Self::read_u64(file)?
            };
            let is_record = if !dim_ids.is_empty() {
                dims[dim_ids[0]].len == 0
            } else {
                false
            };
            vars.push(NetcdfVar {
                name,
                dim_ids,
                vartype,
                vsize,
                begin,
                is_record,
            });
        }
        Ok(vars)
    }

    /// An absent list is encoded as ZERO ZERO; consume and check the count.
    fn read_absent_count(file: &mut File) -> Result<()> {
        if Self::read_u32(file)? != 0 {
            return Err("malformed NetCDF header: absent list has a non-zero count".into());
        }
        Ok(())
    }

    fn read_u32(file: &mut File) -> Result<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    fn read_u64(file: &mut File) -> Result<u64> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        Ok(u64::from_be_bytes(buf))
    }

    fn read_string(file: &mut File) -> Result<String> {
        let len = Self::read_u32(file)? as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        let padded = Self::pad4(len as u64) - len as u64;
        if padded > 0 {
            file.seek(SeekFrom::Current(padded as i64))?;
        }
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    fn type_size(typ: u32) -> Result<usize> {
        match typ {
            1 | 2 => Ok(1),
            3 => Ok(2),
            4 | 5 => Ok(4),
            6 => Ok(8),
            _ => Err("unknown NetCDF type".into()),
        }
    }

    fn pad4(n: u64) -> u64 {
        let rem = n % 4;
        if rem == 0 { n } else { n + (4 - rem) }
    }
}

#[derive(Debug, Clone)]
struct Bat {
    root: [usize; 3],
    torsions: Vec<[usize; 4]>,
    angles: Vec<[usize; 3]>,
}

fn sort_atoms_by_mass(atoms: &[usize], masses: &[f64], reverse: bool) -> Vec<usize> {
    let mut list = atoms.to_vec();
    list.sort_by(|&a, &b| {
        let ma = masses[a];
        let mb = masses[b];
        if ma == mb {
            if reverse { b.cmp(&a) } else { a.cmp(&b) }
        } else if reverse {
            mb.partial_cmp(&ma).unwrap()
        } else {
            ma.partial_cmp(&mb).unwrap()
        }
    });
    list
}

fn build_bat(fragment: &[usize], adjacency: &[Vec<usize>], masses: &[f64]) -> Result<Bat> {
    let fragment_set: HashSet<usize> = fragment.iter().copied().collect();
    if fragment.len() < 3 {
        return Err("fragment must have at least 3 atoms".into());
    }

    let mut terminal_atoms = Vec::new();
    for &a in fragment {
        let degree = adjacency[a]
            .iter()
            .filter(|n| fragment_set.contains(n))
            .count();
        if degree == 1 {
            terminal_atoms.push(a);
        }
    }
    if terminal_atoms.is_empty() {
        return Err("no terminal atoms found for BAT root".into());
    }

    let terminal_sorted = sort_atoms_by_mass(&terminal_atoms, masses, true);
    let initial_atom = terminal_sorted[0];

    let second_atom = adjacency[initial_atom]
        .iter()
        .find(|n| fragment_set.contains(n))
        .copied()
        .ok_or("initial atom has no bonded atom")?;

    let mut third_candidates: Vec<usize> = adjacency[second_atom]
        .iter()
        .filter(|n| **n != initial_atom && fragment_set.contains(n))
        .copied()
        .collect();
    if fragment.len() != 3 {
        let terminal_set: HashSet<usize> = terminal_atoms.into_iter().collect();
        third_candidates.retain(|a| !terminal_set.contains(a));
    }
    if third_candidates.is_empty() {
        return Err("no valid third atom for BAT root".into());
    }
    let third_sorted = sort_atoms_by_mass(&third_candidates, masses, true);
    let third_atom = third_sorted[0];

    let root = [initial_atom, second_atom, third_atom];
    let mut selected_atoms = vec![root[0], root[1], root[2]];
    let mut torsions = Vec::new();

    while selected_atoms.len() < fragment.len() {
        let mut torsion_added = false;
        let mut idx = 0usize;
        while idx < selected_atoms.len() {
            let a1 = selected_atoms[idx];
            let a0_list: Vec<usize> = adjacency[a1]
                .iter()
                .filter(|n| fragment_set.contains(n) && !selected_atoms.contains(n))
                .copied()
                .collect();
            let a0_sorted = sort_atoms_by_mass(&a0_list, masses, false);
            for &a0 in a0_sorted.iter() {
                let a2_list: Vec<usize> = adjacency[a1]
                    .iter()
                    .filter(|n| {
                        **n != a0
                            && fragment_set.contains(n)
                            && selected_atoms.contains(n)
                            && adjacency[**n]
                                .iter()
                                .filter(|m| fragment_set.contains(m))
                                .count()
                                > 1
                    })
                    .copied()
                    .collect();
                let a2_sorted = sort_atoms_by_mass(&a2_list, masses, false);
                if let Some(&a2) = a2_sorted.first() {
                    let a3_list: Vec<usize> = adjacency[a2]
                        .iter()
                        .filter(|n| {
                            **n != a1 && fragment_set.contains(n) && selected_atoms.contains(n)
                        })
                        .copied()
                        .collect();
                    let a3_sorted = sort_atoms_by_mass(&a3_list, masses, false);
                    if let Some(&a3) = a3_sorted.first() {
                        torsions.push([a0, a1, a2, a3]);
                        selected_atoms.push(a0);
                        torsion_added = true;
                    }
                }
            }
            idx += 1;
        }
        if !torsion_added {
            return Err("BAT torsion search failed".into());
        }
    }

    let mut angles = Vec::with_capacity(torsions.len());
    for t in torsions.iter() {
        angles.push([t[0], t[1], t[2]]);
    }

    Ok(Bat {
        root,
        torsions,
        angles,
    })
}

/// What the canonical builder needs to know about the atoms: PDB-style names
/// and elements. Both may be empty, in which case the name-based rules fall
/// back to atom order and the construction is still deterministic.
pub(crate) struct Naming<'a> {
    names: &'a [String],
    atomic_numbers: &'a [i64],
    masses: &'a [f64],
}

/// The backbone atom that continues the chain after `name`, so that walking
/// N -> CA -> C -> N makes the primary torsion about each backbone bond the
/// one that torsion is named for.
fn backbone_successor(name: &str) -> Option<&'static str> {
    match name {
        "N" => Some("CA"),
        "CA" => Some("C"),
        "C" => Some("N"),
        _ => None,
    }
}

impl Naming<'_> {
    fn name(&self, atom: usize) -> &str {
        self.names.get(atom).map_or("", String::as_str)
    }

    /// Atomic number, from the topology when it has one and from the mass
    /// otherwise.
    fn element(&self, atom: usize) -> i64 {
        match self.atomic_numbers.get(atom) {
            Some(&z) if z > 0 => z,
            _ => (self.masses[atom] / 1.9).round().max(1.0) as i64,
        }
    }

    fn is_hydrogen(&self, atom: usize) -> bool {
        self.element(atom) == 1
    }

    /// The trailing digits of a PDB atom name (`CG1` -> 1, `CG` -> 0), which
    /// order the branches of a symmetric centre the way IUPAC does.
    fn numeral(&self, atom: usize) -> u32 {
        let digits: String = self
            .name(atom)
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits
            .chars()
            .rev()
            .collect::<String>()
            .parse()
            .unwrap_or(0)
    }

    /// How good a continuation `child` is from `parent`, smallest first:
    /// the backbone successor, then any other backbone atom, then the heavy
    /// atom IUPAC numbers first (OG1 before CG2, CG1 before CG2), then
    /// hydrogens.
    fn priority(&self, parent: usize, child: usize) -> (u8, i64, u32, usize) {
        let class = if self.is_hydrogen(child) {
            3
        } else if backbone_successor(self.name(parent)) == Some(self.name(child)) {
            0
        } else if matches!(self.name(child), "N" | "CA" | "C") {
            1
        } else {
            2
        };
        (class, -self.element(child), self.numeral(child), child)
    }
}

/// The canonical spanning tree: rooted at the N-terminal backbone nitrogen,
/// grown along the chain, with the chemically named torsion primary about
/// every central bond.
///
/// Walking N -> CA -> C -> N and taking the grandparent's own parent as the
/// reference atom makes the four atoms of each backbone torsion the four the
/// torsion is named for: `C(i-1)-N-CA-C` is phi, `N-CA-C-N(i+1)` is psi,
/// `CA(i-1)-C(i-1)-N-CA` is omega, and in the side chain `N-CA-CB-CG` is
/// chi1, `CA-CB-CG-CD` chi2, and so on outwards.
///
/// Molecules with no protein backbone — a ligand, a small molecule — are
/// rooted at the highest-degree atom and the rest of the rule is unchanged,
/// so the construction stays deterministic and never looks at the trajectory.
fn build_bat_canonical(
    fragment: &[usize],
    adjacency: &[Vec<usize>],
    naming: &Naming,
) -> Result<Bat> {
    let in_fragment: HashSet<usize> = fragment.iter().copied().collect();
    if fragment.len() < 3 {
        return Err("fragment must have at least 3 atoms".into());
    }
    let neighbors = |a: usize| -> Vec<usize> {
        let mut list: Vec<usize> = adjacency[a]
            .iter()
            .copied()
            .filter(|n| in_fragment.contains(n))
            .collect();
        list.sort_unstable();
        list.dedup();
        list
    };
    let ranked = |a: usize| -> Vec<usize> {
        let mut list = neighbors(a);
        list.sort_by_key(|&n| naming.priority(a, n));
        list
    };

    // The N-terminal backbone nitrogen: an N bonded to a CA that is bonded to
    // a C. The lowest index among them is the first residue, since Amber
    // writes residues in order.
    let root = fragment
        .iter()
        .copied()
        .filter(|&a| naming.name(a) == "N")
        .filter(|&a| {
            neighbors(a).into_iter().any(|ca| {
                naming.name(ca) == "CA" && neighbors(ca).into_iter().any(|c| naming.name(c) == "C")
            })
        })
        .min()
        .unwrap_or_else(|| {
            // No backbone: the highest-degree atom, lowest index breaking ties.
            *fragment
                .iter()
                .max_by_key(|&&a| (neighbors(a).len(), usize::MAX - a))
                .expect("non-empty fragment")
        });

    // Breadth-first, each atom's neighbours in canonical order so that ring
    // closures are broken the same way every time.
    let mut parent: HashMap<usize, usize> = HashMap::new();
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut visit_order = vec![root];
    let mut seen: HashSet<usize> = HashSet::from([root]);
    let mut queue = VecDeque::from([root]);
    while let Some(a) = queue.pop_front() {
        for child in ranked(a) {
            if seen.insert(child) {
                parent.insert(child, a);
                children.entry(a).or_default().push(child);
                visit_order.push(child);
                queue.push_back(child);
            }
        }
    }
    if visit_order.len() != fragment.len() {
        return Err("fragment not connected".into());
    }

    // The root triple is the path root -> first -> second, the start of the
    // chain. `first` must have a child of its own for the path to exist.
    let root_children = children.get(&root).cloned().unwrap_or_default();
    let first = root_children
        .iter()
        .copied()
        .find(|c| children.contains_key(c))
        .or_else(|| root_children.first().copied())
        .ok_or("fragment has no second atom")?;
    let second = children
        .get(&first)
        .and_then(|list| list.first().copied())
        .or_else(|| root_children.iter().copied().find(|&c| c != first))
        .ok_or("fragment has no third atom for the BAT root")?;
    // Either root -> first -> second, or, when the third atom hangs off the
    // root instead, second -> root -> first. Both are paths of bonded atoms.
    let bat_root = if parent.get(&second) == Some(&first) {
        [root, first, second]
    } else {
        [second, root, first]
    };

    // Coordinates are built in this order: the root triple, then the rest.
    // An atom's reference may be any atom placed before it.
    let mut placement: Vec<usize> = bat_root.to_vec();
    placement.extend(
        visit_order
            .iter()
            .copied()
            .filter(|a| !bat_root.contains(a)),
    );
    let placed_at: HashMap<usize, usize> =
        placement.iter().enumerate().map(|(i, &a)| (a, i)).collect();

    // Subtree sizes break ties in favour of the branch that carries the rest
    // of the molecule, when the names say nothing.
    let mut subtree: HashMap<usize, usize> = visit_order.iter().map(|&a| (a, 1usize)).collect();
    for &a in visit_order.iter().rev() {
        if let Some(&p) = parent.get(&a) {
            let size = subtree[&a];
            *subtree.get_mut(&p).expect("parent is in the tree") += size;
        }
    }

    // Every atom outside the root triple contributes one torsion
    // (child, parent, grandparent, reference), grouped by the central bond.
    let mut grouped: Vec<((usize, usize), Vec<[usize; 4]>)> = Vec::new();
    let mut slot_of: HashMap<(usize, usize), usize> = HashMap::new();
    for &a in placement.iter().skip(bat_root.len()) {
        let p = parent[&a];
        // The root's other children have no grandparent of their own, so they
        // use `first`, which the root triple has already placed.
        let g = parent.get(&p).copied().unwrap_or(first);
        let before = |x: usize| placed_at[&x] < placed_at[&a];
        let reference = parent
            .get(&g)
            .copied()
            .filter(|&h| h != p && h != a && before(h))
            .or_else(|| {
                ranked(g)
                    .into_iter()
                    .find(|&h| h != p && h != a && before(h))
            })
            .or_else(|| {
                placement[..placed_at[&a]]
                    .iter()
                    .copied()
                    .find(|&h| h != p && h != g && h != a)
            })
            .ok_or("no reference atom for a BAT torsion")?;
        // The crate groups phase angles by the unordered central bond, so the
        // grouping here has to match or the primary would not be the one this
        // rule chose.
        let key = (p.min(g), p.max(g));
        let slot = *slot_of.entry(key).or_insert_with(|| {
            grouped.push((key, Vec::new()));
            grouped.len() - 1
        });
        grouped[slot].1.push([a, p, g, reference]);
    }

    // Within each central bond the named torsion goes first: it becomes the
    // primary, and every other member of the group becomes a phase angle
    // measured from it.
    let mut torsions = Vec::new();
    for (_, mut group) in grouped {
        group.sort_by_key(|t| {
            let (class, element, numeral, atom) = naming.priority(t[1], t[0]);
            (class, usize::MAX - subtree[&t[0]], element, numeral, atom)
        });
        torsions.extend(group);
    }

    let angles = torsions.iter().map(|t| [t[0], t[1], t[2]]).collect();
    Ok(Bat {
        root: bat_root,
        torsions,
        angles,
    })
}

fn build_bat_list(
    fragment: &[usize],
    adjacency: &[Vec<usize>],
    hydrogens: &HashSet<usize>,
    masses: &[f64],
    bonds: BondSet,
    tree: TreeRule,
    naming: &Naming,
) -> Result<Vec<Vec<usize>>> {
    let mut bat_list = Vec::new();

    let bat = match tree {
        TreeRule::MassOrdered => build_bat(fragment, adjacency, masses)?,
        TreeRule::Canonical => build_bat_canonical(fragment, adjacency, naming)?,
    };
    if bonds.is_tree() {
        // The two root bonds, then the bond that places each later atom, in
        // the same order as the angles and torsions below. Under
        // `TreeUnconstrained` a bond to hydrogen is held fixed by the
        // simulation and is not a coordinate; the tree itself, and so every
        // angle and torsion, is unchanged.
        let keep = |a: usize, b: usize| {
            bonds == BondSet::Tree || !(hydrogens.contains(&a) || hydrogens.contains(&b))
        };
        if keep(bat.root[0], bat.root[1]) {
            bat_list.push(vec![bat.root[0], bat.root[1]]);
        }
        if keep(bat.root[1], bat.root[2]) {
            bat_list.push(vec![bat.root[1], bat.root[2]]);
        }
        for t in bat.torsions.iter() {
            if keep(t[0], t[1]) {
                bat_list.push(vec![t[0], t[1]]);
            }
        }
    } else {
        let fragment_set: HashSet<usize> = fragment.iter().copied().collect();
        for &a in fragment {
            for &b in adjacency[a].iter() {
                if a < b
                    && fragment_set.contains(&b)
                    && !hydrogens.contains(&a)
                    && !hydrogens.contains(&b)
                {
                    bat_list.push(vec![a, b]);
                }
            }
        }
        // In atom-index order, as MDAnalysis lists them for the original
        // Python implementation. The order only affects rounding in the MIE
        // sums, but keeping it lets the legacy mode match bit for
        // bit.
        bat_list.sort_unstable();
    }

    bat_list.push(vec![bat.root[0], bat.root[1], bat.root[2]]);
    for angle in bat.angles.iter() {
        bat_list.push(vec![angle[0], angle[1], angle[2]]);
    }
    for torsion in bat.torsions.iter() {
        bat_list.push(vec![torsion[0], torsion[1], torsion[2], torsion[3]]);
    }

    Ok(bat_list)
}

/// Which bonds the internal-coordinate set includes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BondSet {
    /// Every bond between two heavy atoms, ring closures included. Bonds to
    /// hydrogen are left out because they are usually constrained. With
    /// [`CoordinateOptions::drop_constrained`], bonds to hydrogen are included
    /// and only the constrained ones are left out.
    #[default]
    HeavyAtom,
    /// The bonds of the BAT spanning tree: the two root bonds and the bond
    /// that places each later atom, hydrogens included, ring closures left out.
    /// With the angles and torsions this is a non-redundant set of 3N - 6
    /// coordinates per molecule, the set the BAT Jacobian is defined for.
    Tree,
    /// The bonds of the BAT spanning tree, minus the bonds to hydrogen, which
    /// SHAKE and the other constraint algorithms hold at a fixed length.
    ///
    /// The spanning tree is built exactly as for [`BondSet::Tree`], so the
    /// angles and torsions are the same and the set is still non-redundant;
    /// only the constrained bond lengths are left out. This is the right set
    /// for a trajectory run with constrained X-H bonds: those lengths carry
    /// no entropy, and keeping them adds the entropy of the constraint
    /// algorithm's numerical noise, a few tenths of k_B per bond. It differs
    /// from [`BondSet::HeavyAtom`], which also drops the ring-closure bonds
    /// and does not follow the tree.
    TreeUnconstrained,
}

impl BondSet {
    /// Whether the bonds are taken from the BAT spanning tree, so that the
    /// coordinates are non-redundant and the Jacobian is defined.
    pub fn is_tree(self) -> bool {
        matches!(self, Self::Tree | Self::TreeUnconstrained)
    }
}

/// Which spanning tree defines the BAT coordinates.
///
/// A molecule has many valid BAT trees, and they do not all give the same
/// answer: the joint entropy is invariant under the phase-angle shear, but a
/// sum of marginals is not, and a second-order truncation is not. On the
/// alanine dipeptide the choice of tree moves a truncated estimate by up to
/// 0.6 kcal/mol, so it is a stated convention, not an implementation detail.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TreeRule {
    /// Rooted at the heaviest terminal atom and grown lightest-first, the
    /// construction MDAnalysis uses. Depends on nothing but the masses, and
    /// makes no attempt to put the chemically named torsion first.
    #[default]
    MassOrdered,
    /// Rooted at the N-terminal backbone nitrogen and grown along the chain,
    /// with the chemically named torsion primary about every central bond:
    /// `C(i-1)-N-CA-C` (phi), `N-CA-C-N(i+1)` (psi), `CA(i-1)-C(i-1)-N-CA`
    /// (omega) and `N-CA-CB-CG` (chi1), with the side chain continuing
    /// outwards in IUPAC order.
    ///
    /// The point is comparability: the primary torsions are the ones people
    /// already name, so the soft set is exactly the named rotatable torsions
    /// and a per-residue map means the same thing in two different proteins.
    /// It also removes the main source of per-atom instability, a hydrogen
    /// that is a primary-torsion partner under one tree and only a
    /// phase-angle partner under another.
    ///
    /// Molecules with no backbone, such as a ligand, are rooted at the
    /// highest-degree atom; everything else about the rule is unchanged, so it
    /// stays deterministic and never looks at the trajectory.
    Canonical,
}

/// How angles and torsions are evaluated from positions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Geometry {
    /// Angles as atan2(|v1 x v2|, v1 . v2), accurate over the whole range and
    /// never NaN for a straight angle; torsions with a correctly rounded
    /// square root. See [`geometry::bond_angle`].
    #[default]
    Standard,
    /// The arithmetic of the original Python implementation, so its
    /// coordinates are reproduced bit for bit: angles as acos of the
    /// normalized dot product, which differs by about 1e-8 rad and can give
    /// NaN for a straight angle, and torsions with NumPy's `pow(x, 0.5)` in
    /// place of `sqrt(x)`. See [`geometry::legacy_bond_angle`] and
    /// [`geometry::legacy_torsion_angle`]. A developer tool for comparison
    /// with historical results, hidden from the documentation.
    #[doc(hidden)]
    Legacy,
}

/// How a topology's internal coordinates are defined.
///
/// The default is heavy-atom bonds, phase angles and the standard geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinateOptions {
    /// Which bonds are coordinates.
    pub bonds: BondSet,
    /// Replace each torsion that shares its central bond with an earlier
    /// torsion by its offset from that earlier one, a phase angle. See
    /// [`InternalCoordinates::phase_references`]. On by default: the raw
    /// torsions of atoms that rotate together, such as a methyl group's
    /// hydrogens, are redundant, and every estimator does worse on them.
    pub phase_angles: bool,
    /// Which spanning tree the coordinates are built on.
    pub tree: TreeRule,
    /// The angle and torsion formulas.
    pub geometry: Geometry,
    /// Drop every bond and angle that the trajectory shows to be constrained:
    /// one whose standard deviation over the frames read is below
    /// [`CONSTRAINT_TOLERANCE`] times its mean. With the heavy-atom bond set,
    /// bonds to hydrogen are then included and dropped only if constrained, so
    /// a trajectory without SHAKE keeps them. A constrained coordinate has no
    /// entropy, and its near-zero spread would make any estimate of it
    /// meaningless. Torsions are never dropped. Off by default.
    pub drop_constrained: bool,
}

/// The relative spread below which [`CoordinateOptions::drop_constrained`]
/// treats a bond or angle as constrained. Constrained bonds to hydrogen spread
/// by at most 1e-5 of their length (Amber SHAKE, OpenMM constraints), free
/// bonds and angles by about 2e-2 at 300 K; the free spread scales with the
/// square root of the temperature, so it stays above the tolerance down to a
/// few kelvin.
pub const CONSTRAINT_TOLERANCE: f64 = 1e-3;

impl Default for CoordinateOptions {
    fn default() -> Self {
        Self {
            bonds: BondSet::default(),
            phase_angles: true,
            tree: TreeRule::default(),
            geometry: Geometry::default(),
            drop_constrained: false,
        }
    }
}

/// What kind of internal coordinate a value is; see
/// [`InternalCoordinates::coordinate_kinds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateKind {
    /// A bond length.
    Bond,
    /// A bond angle.
    Angle,
    /// A torsion.
    Torsion,
    /// A torsion's offset from the torsion at index `reference`, which turns
    /// about the same central bond.
    PhaseAngle {
        /// Index of the reference torsion among the coordinates.
        reference: usize,
    },
}

impl CoordinateKind {
    /// "bond", "angle", "torsion" or "phase angle".
    pub fn name(self) -> &'static str {
        match self {
            Self::Bond => "bond",
            Self::Angle => "angle",
            Self::Torsion => "torsion",
            Self::PhaseAngle { .. } => "phase angle",
        }
    }
}

/// The internal coordinates of the solute of a topology, and their values on
/// the frames of a trajectory.
///
/// Build it from a topology with [`new`](Self::new) or
/// [`with_options`](Self::with_options), then read a trajectory with
/// [`calculate_internal_coords`](Self::calculate_internal_coords).
pub struct InternalCoordinates {
    /// BAT entries, indexing into `atoms` (not into the topology).
    bat_list: Vec<Vec<usize>>,
    /// Topology indices of the atoms the BAT list uses, sorted ascending.
    atoms: Vec<usize>,
    /// Total atom count declared by the topology.
    natom: usize,
    options: CoordinateOptions,
    /// Whether `calculate_internal_coords` dropped every coordinate but the
    /// torsions.
    torsions_only: bool,
    /// The coordinates `calculate_internal_coords` dropped as constrained, as
    /// BAT entries indexing into `atoms`.
    constrained: Vec<Vec<usize>>,
    /// Per frame, the ln J terms of the dropped coordinates, which
    /// [`log_jacobian`](Self::log_jacobian) still includes.
    constrained_log_jacobian: Vec<f64>,
    /// Solute bonds that lie on a ring, as sorted pairs of local indices.
    ring_bonds: Bonds,
    /// Non-ring bonds one end of which carries only two or more terminal atoms
    /// of one element (CH3, NH3+, CF3, ...), as sorted pairs of local indices.
    symmetric_rotor_bonds: Bonds,
    /// Number of coordinates.
    pub dim: usize,
    /// Coordinate values, `int_coords[frame][coordinate]`, filled by
    /// [`calculate_internal_coords`](Self::calculate_internal_coords). Bond
    /// lengths are in the trajectory's units (Å for Amber), angles and
    /// torsions in radians, torsions in (-pi, pi].
    pub int_coords: Vec<Vec<f64>>,
    /// Every coordinate index pair `(i, j)` with `i < j`, filled by
    /// [`coordinate_pairs`](Self::coordinate_pairs). The estimators do not use
    /// it.
    pub pairs: Vec<(usize, usize)>,
}

/// A coordinate's term in ln J: 2 ln b for a bond, ln sin(theta) for an angle,
/// 0 for a torsion.
fn log_jacobian_term(entry: &[usize], value: f64) -> f64 {
    match entry.len() {
        2 => 2.0 * value.ln(),
        3 => value.sin().ln(),
        _ => 0.0,
    }
}

/// Amber atom types that belong only to water models (including the extra
/// point of four-point models such as TIP4P-Ew and OPC).
const WATER_ATOM_TYPES: [&str; 4] = ["OW", "HW", "EP", "EPW"];

/// Residue names used for water by Amber, CHARMM and GROMACS topologies.
const WATER_RESIDUES: [&str; 16] = [
    "WAT", "HOH", "H2O", "SOL", "TIP3", "TP3", "T3P", "TIP4", "TP4", "T4P", "T4E", "TIP5", "TP5",
    "SPC", "SPCE", "OPC",
];

/// Whether atom `index` is excluded before molecules are assembled: water, by
/// type or residue name, and massless virtual sites, which carry no degrees of
/// freedom of their own.
fn is_excluded(prmtop: &Prmtop, index: usize) -> bool {
    let atom_type = prmtop.atom_types[index].trim();
    if WATER_ATOM_TYPES.contains(&atom_type) {
        return true;
    }
    if let Some(label) = prmtop.residue_labels.get(index) {
        let label = label.trim().to_ascii_uppercase();
        if WATER_RESIDUES.contains(&label.as_str()) {
            return true;
        }
    }
    let massless = prmtop.masses[index] <= 0.0;
    let no_element = prmtop.atomic_numbers.get(index) == Some(&0);
    massless || no_element
}

impl InternalCoordinates {
    /// The coordinates of the solute of the topology at `top`, with the
    /// default [`CoordinateOptions`]: heavy-atom bonds, phase angles and the
    /// standard geometry.
    pub fn new(top: &Path) -> Result<Self> {
        Self::with_options(top, CoordinateOptions::default())
    }

    /// The coordinates of the solute of the topology at `top`, defined by
    /// `options`. Fails if the topology cannot be read or a molecule has no
    /// valid BAT tree.
    pub fn with_options(top: &Path, options: CoordinateOptions) -> Result<Self> {
        let prmtop = parse_prmtop(top)?;
        if prmtop.masses.len() != prmtop.natom || prmtop.atom_types.len() != prmtop.natom {
            return Err("topology MASS or atom type section is shorter than NATOM".into());
        }
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); prmtop.natom];
        for (a, b) in prmtop.bonds.iter().copied() {
            if a >= prmtop.natom || b >= prmtop.natom {
                return Err(format!("bond ({a}, {b}) references an atom beyond NATOM").into());
            }
            adjacency[a].push(b);
            adjacency[b].push(a);
        }

        let mut molecule_mask = vec![true; prmtop.natom];
        let mut hydrogens = HashSet::new();
        for (i, typ) in prmtop.atom_types.iter().enumerate() {
            let t = typ.trim();
            if is_excluded(&prmtop, i) {
                molecule_mask[i] = false;
            }
            // With drop_constrained, the trajectory decides which bonds to
            // hydrogen go.
            let hydrogen = (t.starts_with('H') || t.starts_with('h')) && t != "HW";
            if hydrogen && !options.drop_constrained {
                hydrogens.insert(i);
            }
        }

        let mut visited = vec![false; prmtop.natom];
        let mut bat_list = Vec::new();
        let mut atoms = Vec::new();
        for i in 0..prmtop.natom {
            if !molecule_mask[i] || visited[i] {
                continue;
            }
            let mut queue = VecDeque::new();
            let mut fragment = Vec::new();
            queue.push_back(i);
            visited[i] = true;
            while let Some(a) = queue.pop_front() {
                fragment.push(a);
                for &b in adjacency[a].iter() {
                    if molecule_mask[b] && !visited[b] {
                        visited[b] = true;
                        queue.push_back(b);
                    }
                }
            }

            match fragment.len() {
                // A lone atom (a monatomic ion, say) has no internal
                // coordinates at all.
                1 => continue,
                // A diatomic has a single internal coordinate, its bond, which
                // is kept under the same rule as every other bond.
                2 => {
                    let (a, b) = (fragment[0].min(fragment[1]), fragment[0].max(fragment[1]));
                    if options.bonds == BondSet::Tree
                        || (!hydrogens.contains(&a) && !hydrogens.contains(&b))
                    {
                        bat_list.push(vec![a, b]);
                    }
                }
                _ => bat_list.extend(build_bat_list(
                    &fragment,
                    &adjacency,
                    &hydrogens,
                    &prmtop.masses,
                    options.bonds,
                    options.tree,
                    &Naming {
                        names: &prmtop.atom_names,
                        atomic_numbers: &prmtop.atomic_numbers,
                        masses: &prmtop.masses,
                    },
                )?),
            }
            atoms.extend(fragment);
        }

        // The trajectory reader gathers exactly `atoms`, in ascending order, so
        // rewrite every BAT entry to index that gathered buffer.
        atoms.sort_unstable();
        let mut local = vec![usize::MAX; prmtop.natom];
        for (position, &atom) in atoms.iter().enumerate() {
            local[atom] = position;
        }
        for entry in bat_list.iter_mut() {
            for atom in entry.iter_mut() {
                *atom = local[*atom];
                debug_assert_ne!(*atom, usize::MAX, "BAT entry uses an unselected atom");
            }
        }

        let (ring_bonds, symmetric_rotor_bonds) =
            bond_topology(&prmtop, &adjacency, &molecule_mask);
        let to_local = |bonds: Bonds| -> Bonds {
            bonds
                .into_iter()
                .map(|(a, b)| (local[a].min(local[b]), local[a].max(local[b])))
                .collect()
        };
        let dim = bat_list.len();
        Ok(InternalCoordinates {
            bat_list,
            atoms,
            natom: prmtop.natom,
            options,
            torsions_only: false,
            constrained: Vec::new(),
            constrained_log_jacobian: Vec::new(),
            ring_bonds: to_local(ring_bonds),
            symmetric_rotor_bonds: to_local(symmetric_rotor_bonds),
            dim,
            int_coords: Vec::new(),
            pairs: Vec::new(),
        })
    }

    /// Reads up to `frames` frames of the NetCDF trajectory at `traj` and
    /// evaluates every coordinate on them, filling
    /// [`int_coords`](Self::int_coords). `usize::MAX` reads every frame.
    ///
    /// `torsions_only` first drops every coordinate but the torsions, for good:
    /// [`dim`](Self::dim) and every per-coordinate method then describe the
    /// torsions alone.
    ///
    /// With [`CoordinateOptions::drop_constrained`], every constrained bond and
    /// angle is then dropped, for good, as with `torsions_only`; see
    /// [`constrained_coordinates`](Self::constrained_coordinates). This needs
    /// at least two frames.
    ///
    /// The trajectory must have exactly as many atoms as the topology. Phase
    /// angles and the chosen geometry are applied here, so calling this again
    /// recomputes everything from the positions.
    pub fn calculate_internal_coords(
        &mut self,
        traj: &Path,
        frames: usize,
        torsions_only: bool,
    ) -> Result<()> {
        let mut reader = NetcdfReader::open(traj, self.natom)?;
        let vartype = reader.coordinates_vartype()?;
        if torsions_only {
            self.bat_list = self
                .bat_list
                .iter()
                .filter(|x| x.len() == 4)
                .cloned()
                .collect();
            self.dim = self.bat_list.len();
            self.torsions_only = true;
        }
        let legacy = self.options.geometry == Geometry::Legacy;
        if vartype == 5 {
            let coords = reader.read_coordinates_f32(frames, &self.atoms)?;
            self.int_coords = internal_coordinates_f32(&self.bat_list, &coords);
            if legacy {
                self.replace_legacy(
                    &coords,
                    geometry::legacy_bond_angle_f32,
                    geometry::legacy_torsion_angle_f32,
                );
            }
        } else {
            let coords = reader.read_coordinates_f64(frames, &self.atoms)?;
            self.int_coords = internal_coordinates(&self.bat_list, &coords);
            if legacy {
                self.replace_legacy(
                    &coords,
                    geometry::legacy_bond_angle,
                    geometry::legacy_torsion_angle,
                );
            }
        }
        if self.options.drop_constrained {
            self.drop_constrained()?;
        }
        if self.options.phase_angles {
            let references = self.phase_references();
            for frame in self.int_coords.iter_mut() {
                for (index, reference) in references.iter().enumerate() {
                    if let Some(reference) = *reference {
                        frame[index] = wrap_angle(frame[index] - frame[reference]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Drops the bonds and angles whose relative spread over the frames is
    /// below [`CONSTRAINT_TOLERANCE`], keeping their ln J terms.
    fn drop_constrained(&mut self) -> Result<()> {
        let frames = self.int_coords.len();
        if frames < 2 {
            return Err(format!(
                "finding constrained coordinates needs at least 2 frames, got {frames}"
            )
            .into());
        }
        let n = frames as f64;
        let constrained: Vec<bool> = (0..self.dim)
            .map(|c| {
                if self.bat_list[c].len() == 4 {
                    return false;
                }
                let mean = self.int_coords.iter().map(|f| f[c]).sum::<f64>() / n;
                let variance = self
                    .int_coords
                    .iter()
                    .map(|f| (f[c] - mean).powi(2))
                    .sum::<f64>()
                    / n;
                variance.sqrt() < CONSTRAINT_TOLERANCE * mean.abs()
            })
            .collect();
        self.constrained_log_jacobian = self
            .int_coords
            .iter()
            .map(|frame| {
                self.bat_list
                    .iter()
                    .zip(frame)
                    .zip(&constrained)
                    .filter(|(_, dropped)| **dropped)
                    .map(|((entry, &value), _)| log_jacobian_term(entry, value))
                    .sum()
            })
            .collect();
        for frame in self.int_coords.iter_mut() {
            let mut dropped = constrained.iter();
            frame.retain(|_| !dropped.next().unwrap());
        }
        let mut dropped = constrained.iter();
        let (constrained, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.bat_list)
            .into_iter()
            .partition(|_| *dropped.next().unwrap());
        self.bat_list = kept;
        self.constrained = constrained;
        self.dim = self.bat_list.len();
        Ok(())
    }

    /// Topology atom indices of each coordinate dropped as constrained (see
    /// [`CoordinateOptions::drop_constrained`]), in the layout of
    /// [`coordinate_atoms`](Self::coordinate_atoms). Empty until a trajectory
    /// is read, and always empty without `drop_constrained`.
    pub fn constrained_coordinates(&self) -> Vec<Vec<usize>> {
        self.constrained
            .iter()
            .map(|entry| entry.iter().map(|&local| self.atoms[local]).collect())
            .collect()
    }

    /// Re-evaluates every angle and torsion with the legacy formulas.
    fn replace_legacy<T, A, D>(&mut self, coords: &[Vec<[T; 3]>], angle: A, torsion: D)
    where
        T: Copy,
        A: Fn([T; 3], [T; 3], [T; 3]) -> f64,
        D: Fn([T; 3], [T; 3], [T; 3], [T; 3]) -> f64,
    {
        for (frame, positions) in self.int_coords.iter_mut().zip(coords) {
            for (value, entry) in frame.iter_mut().zip(&self.bat_list) {
                match entry[..] {
                    [a, b, c] => *value = angle(positions[a], positions[b], positions[c]),
                    [a, b, c, d] => {
                        *value = torsion(positions[a], positions[b], positions[c], positions[d])
                    }
                    _ => {}
                }
            }
        }
    }

    /// The values of every coordinate on frames `start`, `start + stride`, ...
    /// of those read, as `columns[coordinate][frame]`: the layout the
    /// estimators take. Fails if no frames were read, `start` is past the
    /// last of them, `stride` is 0, or there are no coordinates.
    pub fn columns(&self, start: usize, stride: usize) -> Result<Vec<Vec<f64>>> {
        if stride == 0 {
            return Err("stride must be at least 1".into());
        }
        let frames = self.int_coords.len();
        if frames == 0 {
            return Err("no frames were read from the trajectory".into());
        }
        if start >= frames {
            return Err(format!("start {start} is beyond the {frames} frames read").into());
        }
        if self.dim == 0 {
            return Err("no internal coordinates were generated".into());
        }
        let selected: Vec<&Vec<f64>> = self.int_coords.iter().skip(start).step_by(stride).collect();
        Ok((0..self.dim)
            .map(|coordinate| selected.iter().map(|frame| frame[coordinate]).collect())
            .collect())
    }

    /// The primary torsion about each rotatable bond: for every non-ring
    /// central bond, the first torsion about it in coordinate order. With phase
    /// angles these are exactly the coordinates of kind
    /// [`CoordinateKind::Torsion`] off rings; the split estimator's default
    /// soft set.
    pub fn soft_torsions(&self) -> Vec<usize> {
        let mut seen = HashSet::new();
        self.bat_list
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.len() == 4)
            .filter(|(_, entry)| seen.insert(central_bond(entry)))
            .filter(|(_, entry)| !self.ring_bonds.contains(&central_bond(entry)))
            .map(|(index, _)| index)
            .collect()
    }

    /// The soft torsions about symmetric rotors such as methyl groups: their
    /// basins are equivalent, so they define no conformational states.
    pub fn symmetric_rotor_torsions(&self) -> Vec<usize> {
        self.soft_torsions()
            .into_iter()
            .filter(|&i| {
                self.symmetric_rotor_bonds
                    .contains(&central_bond(&self.bat_list[i]))
            })
            .collect()
    }

    /// Torsions about bonds that lie on a ring.
    pub fn ring_torsions(&self) -> Vec<usize> {
        self.bat_list
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.len() == 4 && self.ring_bonds.contains(&central_bond(entry)))
            .map(|(index, _)| index)
            .collect()
    }

    /// The kind of each coordinate, with the reference torsion of each phase
    /// angle.
    pub fn coordinate_kinds(&self) -> Vec<CoordinateKind> {
        self.bat_list
            .iter()
            .zip(self.phase_references())
            .map(|(entry, reference)| match (entry.len(), reference) {
                (2, _) => CoordinateKind::Bond,
                (3, _) => CoordinateKind::Angle,
                (_, Some(reference)) => CoordinateKind::PhaseAngle { reference },
                _ => CoordinateKind::Torsion,
            })
            .collect()
    }

    /// Topology atom indices of each coordinate: 2 for a bond, 3 for an angle
    /// (vertex in the middle), 4 for a torsion about the middle two atoms.
    pub fn coordinate_atoms(&self) -> Vec<Vec<usize>> {
        self.bat_list
            .iter()
            .map(|entry| entry.iter().map(|&local| self.atoms[local]).collect())
            .collect()
    }

    /// For each coordinate, the index of the torsion it is a phase angle of,
    /// or `None` for a coordinate that is not a phase angle.
    ///
    /// With `phase_angles` set, the first torsion about each central bond, in
    /// coordinate order, keeps its value and every later torsion about the same
    /// bond becomes its offset from that first one, wrapped to (-pi, pi]. The
    /// map is a shear on the torus with unit Jacobian, so the joint entropy of
    /// the full set is unchanged. The offsets of atoms that rotate together are
    /// stiff and nearly independent of the reference, so a truncated MIE drops
    /// less correlation than with the raw torsions.
    pub fn phase_references(&self) -> Vec<Option<usize>> {
        let mut references = vec![None; self.bat_list.len()];
        if !self.options.phase_angles {
            return references;
        }
        let mut first_about: HashMap<(usize, usize), usize> = HashMap::new();
        for (index, entry) in self.bat_list.iter().enumerate() {
            if entry.len() != 4 {
                continue;
            }
            let bond = (entry[1].min(entry[2]), entry[1].max(entry[2]));
            match first_about.get(&bond) {
                Some(&reference) => references[index] = Some(reference),
                None => {
                    first_about.insert(bond, index);
                }
            }
        }
        references
    }

    /// ln J for each frame, where J = prod over bonds of b^2 times prod over
    /// angles of sin(theta) is the internal-coordinate part of the
    /// Cartesian-to-BAT Jacobian. Adding the mean over frames to the BAT
    /// entropy gives the entropy of the internal motion in Cartesian
    /// coordinates. Bond lengths are in the trajectory's units.
    ///
    /// Defined only for a non-redundant tree set, `BondSet::Tree` or
    /// `BondSet::TreeUnconstrained`, without `torsions_only`. Phase angles do
    /// not change it.
    ///
    /// The two ways of leaving constrained bonds out differ here. A coordinate
    /// dropped by `drop_constrained` still contributes its term, so
    /// `--tree-bonds --drop-constrained --jacobian` gives the entropy and the
    /// full Jacobian in one run. Under `TreeUnconstrained` the constrained
    /// bonds are never coordinates in the first place, so their `2 ln b` terms
    /// are left out; each is a constant of the molecule and cancels in a
    /// difference between two ensembles of it.
    pub fn log_jacobian(&self) -> Result<Vec<f64>> {
        if !self.options.bonds.is_tree() {
            return Err("the BAT Jacobian needs a tree bond set".into());
        }
        if self.torsions_only {
            return Err("the BAT Jacobian needs bonds and angles, not torsions only".into());
        }
        Ok(self
            .int_coords
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                let constrained = self
                    .constrained_log_jacobian
                    .get(index)
                    .copied()
                    .unwrap_or(0.0);
                self.bat_list
                    .iter()
                    .zip(frame)
                    .map(|(entry, &value)| log_jacobian_term(entry, value))
                    .sum::<f64>()
                    + constrained
            })
            .collect())
    }

    /// Fills [`pairs`](Self::pairs) with every coordinate index pair `(i, j)`,
    /// `i < j`, in lexicographic order, the order of
    /// [`estimate_coordinate_mutual_information`](crate::estimate_coordinate_mutual_information).
    pub fn coordinate_pairs(&mut self) {
        let mut pairs = Vec::new();
        for i in 0..self.dim {
            for j in 0..self.dim {
                if i >= j {
                    continue;
                }
                pairs.push((i, j));
            }
        }
        self.pairs = pairs;
    }

    /// The metric of each coordinate: periodic with period 2 pi for torsions
    /// (phase angles included), linear for bonds and angles. Pass it to the
    /// estimators through [`EntropyOptions::metrics`](crate::EntropyOptions::metrics).
    pub fn coordinate_metrics(&self) -> Vec<CoordinateMetric> {
        self.bat_list
            .iter()
            .map(|entry| {
                if entry.len() == 4 {
                    CoordinateMetric::Periodic {
                        period: 2.0 * std::f64::consts::PI,
                    }
                } else {
                    CoordinateMetric::Linear
                }
            })
            .collect()
    }
}

/// A set of bonds, each a sorted pair of atom indices.
type Bonds = HashSet<(usize, usize)>;

/// The central bond of a torsion entry, as a sorted pair.
fn central_bond(entry: &[usize]) -> (usize, usize) {
    (entry[1].min(entry[2]), entry[1].max(entry[2]))
}

/// Ring bonds and symmetric rotor bonds of the solute, as sorted pairs of
/// topology indices. A bond lies on a ring exactly when it is not a bridge of
/// the bond graph.
fn bond_topology(prmtop: &Prmtop, adjacency: &[Vec<usize>], include: &[bool]) -> (Bonds, Bonds) {
    let n = adjacency.len();
    let neighbors = |a: usize| adjacency[a].iter().copied().filter(move |&b| include[b]);

    // Iterative Tarjan bridge search, so large molecules cannot overflow the
    // stack.
    let mut discovered = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut timer = 0;
    let mut bridges = HashSet::new();
    for root in (0..n).filter(|&a| include[a]) {
        if discovered[root] != usize::MAX {
            continue;
        }
        discovered[root] = timer;
        low[root] = timer;
        timer += 1;
        // (atom, parent, next neighbor to visit)
        let mut stack = vec![(root, usize::MAX, 0usize)];
        while let Some(top) = stack.len().checked_sub(1) {
            let (atom, parent, next) = stack[top];
            if let Some(&neighbor) = adjacency[atom].get(next) {
                stack[top].2 += 1;
                if !include[neighbor] || neighbor == parent {
                    continue;
                }
                if discovered[neighbor] == usize::MAX {
                    discovered[neighbor] = timer;
                    low[neighbor] = timer;
                    timer += 1;
                    stack.push((neighbor, atom, 0));
                } else {
                    low[atom] = low[atom].min(discovered[neighbor]);
                }
            } else {
                stack.pop();
                if let Some(&(up, _, _)) = stack.last() {
                    low[up] = low[up].min(low[atom]);
                    if low[atom] > discovered[up] {
                        bridges.insert((up.min(atom), up.max(atom)));
                    }
                }
            }
        }
    }

    let element = |a: usize| -> u64 {
        match prmtop.atomic_numbers.get(a) {
            Some(&z) => z as u64,
            None => prmtop.masses[a].to_bits(),
        }
    };
    let mut ring_bonds = HashSet::new();
    let mut symmetric_rotor_bonds = HashSet::new();
    for a in (0..n).filter(|&a| include[a]) {
        for b in neighbors(a).filter(|&b| a < b) {
            let bond = (a, b);
            if !bridges.contains(&bond) {
                ring_bonds.insert(bond);
                continue;
            }
            // One end carries only two or more terminal atoms of one element.
            let symmetric_end = |end: usize, other: usize| {
                let substituents: Vec<usize> = neighbors(end).filter(|&x| x != other).collect();
                substituents.len() >= 2
                    && substituents.iter().all(|&x| neighbors(x).count() == 1)
                    && substituents
                        .iter()
                        .all(|&x| element(x) == element(substituents[0]))
            };
            if symmetric_end(a, b) || symmetric_end(b, a) {
                symmetric_rotor_bonds.insert(bond);
            }
        }
    }
    (ring_bonds, symmetric_rotor_bonds)
}

/// `angle` wrapped to (-pi, pi], the range of the torsions themselves.
fn wrap_angle(angle: f64) -> f64 {
    use std::f64::consts::PI;
    PI - (PI - angle).rem_euclid(2.0 * PI)
}
