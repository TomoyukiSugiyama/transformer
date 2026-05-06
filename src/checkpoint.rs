use std::{
    collections::HashMap,
    fs::{File, create_dir_all},
    io::{self, BufReader, BufWriter, Read, Write},
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
    fn from_weight_map(&mut self, map: &WeightMap) -> io::Result<()>;
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

    pub fn insert_strings(&mut self, key: &str, v: Vec<String>) {
        self.strings.insert(key.to_string(), v);
    }

    pub fn get_scalar(&self, key: &str) -> io::Result<u64> {
        self.scalars.get(key).copied().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("missing scalar: {key}"))
        })
    }

    pub fn get_vector(&self, key: &str) -> io::Result<&Vector> {
        self.vectors.get(key).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("missing vector: {key}"))
        })
    }

    pub fn get_matrix(&self, key: &str) -> io::Result<&Matrix> {
        self.matrices.get(key).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("missing matrix: {key}"))
        })
    }

    pub fn get_strings(&self, key: &str) -> io::Result<&Vec<String>> {
        self.strings.get(key).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing strings: {key}"),
            )
        })
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

    pub fn scoped(&self, prefix: &str) -> WeightMap {
        let mut out = WeightMap::new();
        let p = format!("{prefix}.");
        for (k, v) in &self.scalars {
            if let Some(rest) = k.strip_prefix(&p) {
                out.insert_scalar(rest, *v);
            }
        }
        for (k, v) in &self.vectors {
            if let Some(rest) = k.strip_prefix(&p) {
                out.insert_vector(rest, v.clone());
            }
        }
        for (k, v) in &self.matrices {
            if let Some(rest) = k.strip_prefix(&p) {
                out.insert_matrix(rest, v.clone());
            }
        }
        for (k, v) in &self.strings {
            if let Some(rest) = k.strip_prefix(&p) {
                out.insert_strings(rest, v.clone());
            }
        }
        out
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

    pub fn load(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        Self::read_from(&mut r)
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
            write_u64(w, self.strings[key].len() as u64)?;
            for s in &self.strings[key] {
                write_string(w, s)?;
            }
        }
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }
        let mut ver_buf = [0u8; 4];
        r.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported checkpoint version {version}"),
            ));
        }
        let n = read_u64(r)? as usize;
        let mut map = WeightMap::new();
        for _ in 0..n {
            let kind = read_u8(r)?;
            let key = read_string(r)?;
            match kind {
                KIND_SCALAR => {
                    let v = read_u64(r)?;
                    map.insert_scalar(&key, v);
                }
                KIND_VECTOR => {
                    let v = read_vec_f32(r)?;
                    map.insert_vector(&key, v);
                }
                KIND_MATRIX => {
                    let v = read_matrix(r)?;
                    map.insert_matrix(&key, v);
                }
                KIND_STRINGS => {
                    let cnt = read_u64(r)? as usize;
                    let mut v = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        let str = read_string(r)?;
                        v.push(str);
                    }
                    map.insert_strings(&key, v);
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown entry kind: {kind}"),
                    ));
                }
            }
        }

        Ok(map)
    }

    pub fn vector_keys(&self) -> impl Iterator<Item = &str> {
        self.vectors.keys().map(|k| k.as_str())
    }
}

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn write_vec_f32<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    write_u64(w, v.len() as u64)?;
    for &x in v {
        write_f32(w, x)?;
    }
    Ok(())
}

fn read_vec_f32<R: Read>(r: &mut R) -> io::Result<Vec<f32>> {
    let n = read_u64(r)? as usize;
    (0..n).map(|_| read_f32(r)).collect()
}

fn write_matrix<W: Write>(w: &mut W, m: &[Vec<f32>]) -> io::Result<()> {
    write_u64(w, m.len() as u64)?;
    for row in m {
        write_vec_f32(w, row)?;
    }
    Ok(())
}

fn read_matrix<R: Read>(r: &mut R) -> io::Result<Vec<Vec<f32>>> {
    let row = read_u64(r)? as usize;
    (0..row).map(|_| read_vec_f32(r)).collect()
}

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u64(w, bytes.len() as u64)?;

    w.write_all(bytes)
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
