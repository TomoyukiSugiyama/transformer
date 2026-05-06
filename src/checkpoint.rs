use std::{
    collections::HashMap,
    fs::{File, create_dir_all},
    io::{self, BufWriter, Write},
    path::Path,
};

pub type Vector = Vec<f32>;
pub type Matrix = Vec<Vec<f32>>;

const MAGIC: &[u8; 4] = b"TFWM";
const VERSION: u32 = 1;
const KIND_SCALAR: u8 = 1;
const KIND_VECTOR: u8 = 2;
const KIND_MATRIX: u8 = 3;
const KIND_STRINGS: u8 = 4;

pub trait Checkpointable {
    fn to_weight_map(&self) -> WeightMap;
}

#[derive(Default)]
pub struct WeightMap {
    scalars: HashMap<String, u64>,
    vectors: HashMap<String, Vector>,
    matrices: HashMap<String, Matrix>,
    strings: HashMap<String, Vec<String>>,
}

impl WeightMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_scalar(&mut self, key: &str, v: u64) {
        self.scalars.insert(key.to_string(), v);
    }

    pub fn insert_vector(&mut self, key: &str, v: Vector) {
        self.vectors.insert(key.to_string(), v);
    }

    pub fn insert_matrix(&mut self, key: &str, v: Matrix) {
        self.matrices.insert(key.to_string(), v);
    }

    pub fn insert_string(&mut self, key: &str, v: Vec<String>) {
        self.strings.insert(key.to_string(), v);
    }

    pub fn merge(&mut self, prefix: &str, other: WeightMap) {
        for (k, v) in other.scalars {
            self.scalars.insert(format!("{prefix}.{k}"), v);
        }
        for (k, v) in other.vectors {
            self.vectors.insert(format!("{prefix}.{k}"), v);
        }
        for (k, v) in other.matrices {
            self.matrices.insert(format!("{prefix}.{k}"), v);
        }
        for (k, v) in other.strings {
            self.strings.insert(format!("{prefix}.{k}"), v);
        }
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        self.write_to(&mut w)
    }

    /// リトルエンディアンで書き込む
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;

        let total =
            self.scalars.len() + self.vectors.len() + self.matrices.len() + self.strings.len();
        write_u64(w, total as u64)?;

        let mut scalar_keys: Vec<_> = self.scalars.keys().collect();
        scalar_keys.sort();
        for key in scalar_keys {
            write_u8(w, KIND_SCALAR)?;
            write_string(w, key)?;
            write_u64(w, self.scalars[key])?;
        }

        let mut vector_keys: Vec<_> = self.vectors.keys().collect();
        vector_keys.sort();
        for key in vector_keys {
            write_u8(w, KIND_VECTOR)?;
            write_string(w, key)?;
            write_vec_f32(w, &self.vectors[key])?;
        }

        let mut matrix_keys: Vec<_> = self.matrices.keys().collect();
        matrix_keys.sort();
        for key in matrix_keys {
            write_u8(w, KIND_MATRIX)?;
            write_string(w, key)?;
            write_matrix(w, &self.matrices[key])?;
        }

        let mut string_keys: Vec<_> = self.strings.keys().collect();
        string_keys.sort();
        for key in string_keys {
            write_u8(w, KIND_STRINGS)?;
            write_string(w, key)?;
            for s in &self.strings[key] {
                write_string(w, s)?;
            }
        }
        Ok(())
    }
}

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_vec_f32<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    write_u64(w, v.len() as u64)?;
    for &x in v {
        write_f32(w, x)?;
    }
    Ok(())
}

fn write_matrix<W: Write>(w: &mut W, m: &[Vec<f32>]) -> io::Result<()> {
    write_u64(w, m.len() as u64)?;
    for row in m {
        write_vec_f32(w, row)?;
    }
    Ok(())
}

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u64(w, bytes.len() as u64)?;

    w.write_all(bytes)
}
