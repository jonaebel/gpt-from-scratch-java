package network;

import math.Matrix;
import math.MatrixMultiplication;

import java.util.Random;

/**
 * Single-Head Scaled Dot-Product Attention mit Causal Mask.
 *
 * Forward:
 *   Q = X @ W_Q^T,  K = X @ W_K^T,  V = X @ W_V^T     [seqLen × dK]
 *   S = Q @ K^T / sqrt(dK)   — causal mask → softmax    [seqLen × seqLen]
 *   out = A @ V                                          [seqLen × dK]
 *
 * Gradient-Akkumulation: backward() addiert zu den Gewichtsgradienten.
 * Vor jedem Batch: zeroGradients() aufrufen.
 * Nach jedem Batch: adamStep() aufrufen.
 */
public class SelfAttentionLayer {

    private final int embDim;
    private final int dK;

    // Gewichte [dK × embDim] — transponiert gespeichert (multiplyBT-Konvention)
    Matrix W_Q, W_K, W_V;
    Matrix gradW_Q, gradW_K, gradW_V;

    // Adam-Momente (lazy init)
    private double[][] mW_Q, vW_Q;
    private double[][] mW_K, vW_K;
    private double[][] mW_V, vW_V;

    // Zwischengespeichert für backward()
    private Matrix lastX;
    private Matrix lastQ, lastK, lastV;
    private Matrix lastWeights; // softmax(S)

    public SelfAttentionLayer(int embDim, int dK) {
        this.embDim = embDim;
        this.dK     = dK;

        double std = Math.sqrt(2.0 / (embDim + dK));
        W_Q = randomMatrix(dK, embDim, std);
        W_K = randomMatrix(dK, embDim, std);
        W_V = randomMatrix(dK, embDim, std);

        gradW_Q = new Matrix(dK, embDim);
        gradW_K = new Matrix(dK, embDim);
        gradW_V = new Matrix(dK, embDim);
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /** X: [seqLen × embDim] → [seqLen × dK] */
    public Matrix forward(Matrix X) {
        lastX = X;
        lastQ = MatrixMultiplication.multiplyBT(X, W_Q);
        lastK = MatrixMultiplication.multiplyBT(X, W_K);
        lastV = MatrixMultiplication.multiplyBT(X, W_V);

        Matrix scores = MatrixMultiplication.multiply(lastQ, lastK.transpose())
                                            .multiply(1.0 / Math.sqrt(dK));
        applyCausalMask(scores);
        lastWeights = softmaxRows(scores);

        return MatrixMultiplication.multiply(lastWeights, lastV);
    }

    // ── Backward (akkumulierend) ───────────────────────────────────────────────

    /** gradOutput: [seqLen × dK] → dX [seqLen × embDim] */
    public Matrix backward(Matrix gradOutput) {
        // out = A @ V
        Matrix dV = MatrixMultiplication.multiply(lastWeights.transpose(), gradOutput);
        Matrix dA = MatrixMultiplication.multiply(gradOutput, lastV.transpose());

        // Softmax-Backward (voller Jacobian)
        Matrix dScores = softmaxBackward(dA, lastWeights).multiply(1.0 / Math.sqrt(dK));

        // scores = Q @ K^T
        Matrix dQ = MatrixMultiplication.multiply(dScores, lastK);
        Matrix dK = MatrixMultiplication.multiply(dScores.transpose(), lastQ);

        // Gewichtsgradienten akkumulieren (+=)
        gradW_Q = gradW_Q.add(MatrixMultiplication.multiplyATB(dQ, lastX));
        gradW_K = gradW_K.add(MatrixMultiplication.multiplyATB(dK, lastX));
        gradW_V = gradW_V.add(MatrixMultiplication.multiplyATB(dV, lastX));

        // dX = dQ @ W_Q + dK @ W_K + dV @ W_V
        return MatrixMultiplication.multiply(dQ, W_Q)
                 .add(MatrixMultiplication.multiply(dK, W_K))
                 .add(MatrixMultiplication.multiply(dV, W_V));
    }

    // ── Gradient-Management ───────────────────────────────────────────────────

    public void zeroGradients() {
        gradW_Q = new Matrix(dK, embDim);
        gradW_K = new Matrix(dK, embDim);
        gradW_V = new Matrix(dK, embDim);
    }

    // ── Adam ──────────────────────────────────────────────────────────────────

    public void adamStep(double lr, double beta1, double beta2, double eps, int t, int batchSize) {
        if (mW_Q == null) {
            mW_Q = new double[dK][embDim]; vW_Q = new double[dK][embDim];
            mW_K = new double[dK][embDim]; vW_K = new double[dK][embDim];
            mW_V = new double[dK][embDim]; vW_V = new double[dK][embDim];
        }
        double bc1 = 1 - Math.pow(beta1, t);
        double bc2 = 1 - Math.pow(beta2, t);

        adamUpdate(W_Q.getValues(), gradW_Q.getValues(), mW_Q, vW_Q, lr, beta1, beta2, eps, bc1, bc2, batchSize);
        adamUpdate(W_K.getValues(), gradW_K.getValues(), mW_K, vW_K, lr, beta1, beta2, eps, bc1, bc2, batchSize);
        adamUpdate(W_V.getValues(), gradW_V.getValues(), mW_V, vW_V, lr, beta1, beta2, eps, bc1, bc2, batchSize);
    }

    static void adamUpdate(double[][] w, double[][] g, double[][] m, double[][] v,
                           double lr, double beta1, double beta2, double eps,
                           double bc1, double bc2, int batchSize) {
        for (int i = 0; i < w.length; i++)
            for (int j = 0; j < w[0].length; j++) {
                double gi = g[i][j] / batchSize;
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * gi;
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * gi * gi;
                w[i][j] -= lr * (m[i][j] / bc1) / (Math.sqrt(v[i][j] / bc2) + eps);
            }
    }

    // ── Hilfsmethoden ─────────────────────────────────────────────────────────

    private void applyCausalMask(Matrix scores) {
        double[][] v = scores.getValues();
        for (int i = 0; i < v.length; i++)
            for (int j = i + 1; j < v[0].length; j++)
                v[i][j] = Double.NEGATIVE_INFINITY;
    }

    private Matrix softmaxRows(Matrix m) {
        double[][] in  = m.getValues();
        int rows = in.length, cols = in[0].length;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double max = in[i][0];
            for (int j = 1; j < cols; j++) if (in[i][j] > max) max = in[i][j];
            double sum = 0;
            for (int j = 0; j < cols; j++) { out[i][j] = Math.exp(in[i][j] - max); sum += out[i][j]; }
            for (int j = 0; j < cols; j++) out[i][j] /= sum;
        }
        return new Matrix(out);
    }

    private Matrix softmaxBackward(Matrix dOut, Matrix A) {
        double[][] dy = dOut.getValues();
        double[][] a  = A.getValues();
        int rows = a.length, cols = a[0].length;
        double[][] dx = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double dot = 0;
            for (int j = 0; j < cols; j++) dot += a[i][j] * dy[i][j];
            for (int j = 0; j < cols; j++) dx[i][j] = a[i][j] * (dy[i][j] - dot);
        }
        return new Matrix(dx);
    }

    private static Matrix randomMatrix(int rows, int cols, double std) {
        Random rng = new Random();
        double[][] v = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                v[i][j] = rng.nextGaussian() * std;
        return new Matrix(v);
    }

    // ── Getter ────────────────────────────────────────────────────────────────

    public Matrix getW_Q()     { return W_Q; }
    public Matrix getW_K()     { return W_K; }
    public Matrix getW_V()     { return W_V; }
    public Matrix getGradW_Q() { return gradW_Q; }
    public Matrix getGradW_K() { return gradW_K; }
    public Matrix getGradW_V() { return gradW_V; }
    public void setW_Q(Matrix m) { W_Q = m; }
    public void setW_K(Matrix m) { W_K = m; }
    public void setW_V(Matrix m) { W_V = m; }
}
