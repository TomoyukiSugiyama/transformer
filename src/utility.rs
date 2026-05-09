/// (m, k) × (k, n) → (m, n)
pub fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    let (_m, k, n) = (a.len(), b.len(), b[0].len());
    let mut out: Vec<Vec<f32>> = a.iter().map(|_| vec![0.0f32; n]).collect();
    out.par_iter_mut().enumerate().for_each(|(i, out_row)| {
        let a_row = &a[i];
        for l in 0..k {
            let aik = a_row[l];
            let b_row = &b[l];
            for j in 0..n {
                out_row[j] += aik * b_row[j];
            }
        }
    });
    out
}

/// 行列を転置: (m, n) → (n, m)
pub fn transpose(a: &[Vec<f32>]) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    let (m, n) = (a.len(), a[0].len());
    let mut out: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; m]).collect();
    out.par_iter_mut().enumerate().for_each(|(j, out_row)| {
        for i in 0..m {
            out_row[i] = a[i][j];
        }
    });
    out
}

/// Softmax（行ごと、数値安定版）
pub fn softmax_rows(a: &mut Vec<Vec<f32>>) {
    use rayon::prelude::*;
    a.par_iter_mut().for_each(|row| {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
        row.iter_mut().for_each(|x| *x = ((*x) - max).exp() / sum);
    });
}

/// 線形変換: (seq_len, d_in) × W(d_in, d_out) → (seq_len, d_out)
pub fn linear(x: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
    matmul(x, w)
}

/// 行列を in-place で加算: a += b
pub fn add_matrix_in_place(a: &mut [Vec<f32>], b: &[Vec<f32>]) {
    for (a_row, b_row) in a.iter_mut().zip(b.iter()) {
        for (a_v, b_v) in a_row.iter_mut().zip(b_row.iter()) {
            *a_v += *b_v;
        }
    }
}
