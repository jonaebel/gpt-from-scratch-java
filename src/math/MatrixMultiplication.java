package math;

import java.util.stream.IntStream;

public class MatrixMultiplication {

    // Parallelize when total multiply-add ops exceed this threshold.
    // Keeps overhead out of tiny matrices (e.g. output layer, bias loops).
    private static final long PARALLEL_THRESHOLD = 500_000L;

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * C = A (n×k) @ B (k×m).
     * Internally pre-transposes B to BT (m×k) so the inner loop is row-major
     * (cache-friendly). Parallelises over output rows for large matrices.
     */
    public static Matrix multiply(Matrix a, Matrix b) {
        double[][] av = a.values;
        double[][] bv = b.values;
        int n = av.length, k = av[0].length, m = bv[0].length;

        if (k != bv.length)
            throw new IllegalArgumentException(
                "Dimension mismatch: A is " + n + "×" + k +
                " but B is " + bv.length + "×" + m);

        double[][] bt = transposRaw(bv, k, m);
        double[][] result = new double[n][m];
        dispatch(n, k, m, av, bt, result);
        return new Matrix(result);
    }

    /**
     * C = A (n×k) @ BT^T   where BT is stored as (m×k) — already in transposed layout.
     * Zero-allocation hot path: no transposition step needed.
     *
     * Use this when the right-hand matrix is stored as [outSize × inSize],
     * i.e. the weight matrix in Layer.
     */
    public static Matrix multiplyBT(Matrix a, Matrix bt) {
        double[][] av  = a.values;
        double[][] btv = bt.values;
        int n = av.length, k = av[0].length, m = btv.length;

        if (k != btv[0].length)
            throw new IllegalArgumentException(
                "Dimension mismatch: A cols " + k + " ≠ BT cols " + btv[0].length);

        double[][] result = new double[n][m];
        dispatch(n, k, m, av, btv, result);
        return new Matrix(result);
    }

    /**
     * C = A^T (n×k) @ B (k×m)   where A is stored normally as (k×n).
     * Avoids allocating a transposed copy of A by swapping the loop order.
     * Optimised for small k (e.g. batch_size = 64).
     *
     * Use for gradWeights = delta^T @ lastInput, where delta is [batch × outSize].
     */
    public static Matrix multiplyATB(Matrix a, Matrix b) {
        double[][] av = a.values;
        double[][] bv = b.values;
        int k = av.length, n = av[0].length, m = bv[0].length;

        if (k != bv.length)
            throw new IllegalArgumentException(
                "Dimension mismatch: A^T rows " + k + " ≠ B rows " + bv.length);

        double[][] result = new double[n][m];

        if ((long) n * k * m >= PARALLEL_THRESHOLD)
            IntStream.range(0, n).parallel().forEach(i -> atbRow(i, av, bv, result[i], k, m));
        else
            for (int i = 0; i < n; i++) atbRow(i, av, bv, result[i], k, m);

        return new Matrix(result);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /** Parallel or sequential dispatch for C = A @ BT (BT already transposed). */
    private static void dispatch(int n, int k, int m,
                                 double[][] av, double[][] bt, double[][] result) {
        if ((long) n * k * m >= PARALLEL_THRESHOLD)
            IntStream.range(0, n).parallel().forEach(i -> dotRow(av[i], bt, result[i], m, k));
        else
            for (int i = 0; i < n; i++) dotRow(av[i], bt, result[i], m, k);
    }

    /** result[j] = dot(aRow, bt[j])  for j = 0..m-1 */
    private static void dotRow(double[] aRow, double[][] bt, double[] dst, int m, int k) {
        for (int j = 0; j < m; j++) {
            double[] btj = bt[j];
            double s = 0;
            for (int l = 0; l < k; l++) s += aRow[l] * btj[l];
            dst[j] = s;
        }
    }

    /**
     * One output row of A^T @ B (loop-swap: iterate l first so B rows are
     * accessed sequentially — cache-friendly even when k is small).
     * dst must be pre-zeroed (guaranteed by new double[n][m]).
     */
    private static void atbRow(int i, double[][] av, double[][] bv,
                                double[] dst, int k, int m) {
        for (int l = 0; l < k; l++) {
            double ali = av[l][i];      // scalar: one column access, then reused m times
            double[] bl = bv[l];        // row of B — sequential, cache-friendly
            for (int j = 0; j < m; j++) dst[j] += ali * bl[j];
        }
    }

    /** Allocates and fills the transpose of src (rows × cols). */
    private static double[][] transposRaw(double[][] src, int rows, int cols) {
        double[][] t = new double[cols][rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[c][r] = src[r][c];
        return t;
    }
}
