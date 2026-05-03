/// (m, k) × (k, n) → (m, n)
pub fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (m, k, n) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0f32; n]; m];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                out[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    out
}

/// 行列を転置: (m, n) → (n, m)
pub fn transpose(a: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (m, n) = (a.len(), a[0].len());
    let mut out = vec![vec![0.0f32; m]; n];
    for i in 0..m { for j in 0..n { out[j][i] = a[i][j]; } }
    out
}

/// Softmax（行ごと、数値安定版）
pub fn softmax_rows(a: &mut Vec<Vec<f32>>) {
    for row in a.iter_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
        row.iter_mut().for_each(|x| *x = ((*x) - max).exp() / sum);
    }
}

/// 線形変換: (seq_len, d_in) × W(d_in, d_out) → (seq_len, d_out)
pub fn linear(x: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
    matmul(x, w)
}