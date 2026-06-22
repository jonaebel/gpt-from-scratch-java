package network;

import math.Matrix;
import math.MatrixMultiplication;

import java.util.Random;

/**
 * Vollständiger Transformer-Block (Pre-Norm, Decoder-Stil):
 *
 *   Sub-Block 1:  res1    = X    + MHA(LayerNorm(X))
 *   Sub-Block 2:  output  = res1 + FFN(LayerNorm(res1))
 *
 * FFN:  ReLU(x @ W1^T + b1) @ W2^T + b2,   ffnDim = 4 * embDim
 *
 * Gradient-Akkumulation: backward() addiert zu allen Gewichtsgradienten.
 * Vor jedem Batch: zeroGradients(). Nach jedem Batch: adamStep().
 */
public class TransformerBlock {

    private final int embDim;
    private final int ffnDim;

    final LayerNorm          norm1, norm2;
    final MultiHeadAttention attention;

    // FFN-Gewichte [outDim × inDim] — transponiert gespeichert
    Matrix   W1;      // [ffnDim × embDim]
    Matrix   W2;      // [embDim  × ffnDim]
    double[] b1;      // [ffnDim]
    double[] b2;      // [embDim]

    Matrix   gradW1;  // [ffnDim × embDim]
    Matrix   gradW2;  // [embDim  × ffnDim]
    double[] gradB1;  // [ffnDim]
    double[] gradB2;  // [embDim]

    // Adam-Momente für FFN (lazy init)
    private double[][] mW1, vW1;
    private double[][] mW2, vW2;
    private double[]   mB1, vB1;
    private double[]   mB2, vB2;

    // Zwischengespeichert für backward()
    private Matrix lastX;
    private Matrix lastNorm1Out;
    private Matrix lastAttnOut;
    private Matrix lastRes1;
    private Matrix lastNorm2Out;
    private Matrix lastFFN1Pre;
    private Matrix lastFFN1;

    public TransformerBlock(int embDim, int numHeads) {
        this.embDim = embDim;
        this.ffnDim = 4 * embDim;

        norm1     = new LayerNorm(embDim);
        norm2     = new LayerNorm(embDim);
        attention = new MultiHeadAttention(embDim, numHeads);

        W1 = randomMatrix(ffnDim, embDim, Math.sqrt(2.0 / embDim));
        W2 = randomMatrix(embDim, ffnDim, Math.sqrt(2.0 / ffnDim));
        b1 = new double[ffnDim];
        b2 = new double[embDim];

        gradW1 = new Matrix(ffnDim, embDim);
        gradW2 = new Matrix(embDim, ffnDim);
        gradB1 = new double[ffnDim];
        gradB2 = new double[embDim];
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /** X: [seqLen × embDim] → [seqLen × embDim] */
    public Matrix forward(Matrix X) {
        lastX = X;

        lastNorm1Out = norm1.forward(X);
        lastAttnOut  = attention.forward(lastNorm1Out);
        lastRes1     = X.add(lastAttnOut);

        lastNorm2Out = norm2.forward(lastRes1);
        lastFFN1Pre  = MatrixMultiplication.multiplyBT(lastNorm2Out, W1).addVector(b1);
        lastFFN1     = relu(lastFFN1Pre);
        Matrix ffnOut = MatrixMultiplication.multiplyBT(lastFFN1, W2).addVector(b2);

        return lastRes1.add(ffnOut);
    }

    // ── Backward (akkumulierend) ───────────────────────────────────────────────

    /** gradOutput: [seqLen × embDim] → dX [seqLen × embDim] */
    public Matrix backward(Matrix gradOutput) {
        int seqLen = gradOutput.getRows();

        // ── FFN rückwärts ─────────────────────────────────────────────────────
        // gradW2, gradB2 akkumulieren
        gradW2 = gradW2.add(MatrixMultiplication.multiplyATB(gradOutput, lastFFN1));
        double[][] dfo = gradOutput.getValues();
        for (int j = 0; j < embDim; j++) { double s = 0; for (int i = 0; i < seqLen; i++) s += dfo[i][j]; gradB2[j] += s; }

        Matrix dFFN1     = MatrixMultiplication.multiply(gradOutput, W2);
        Matrix dFFN1Pre  = reluBackward(dFFN1, lastFFN1Pre);

        gradW1 = gradW1.add(MatrixMultiplication.multiplyATB(dFFN1Pre, lastNorm2Out));
        double[][] dfp = dFFN1Pre.getValues();
        for (int j = 0; j < ffnDim; j++) { double s = 0; for (int i = 0; i < seqLen; i++) s += dfp[i][j]; gradB1[j] += s; }

        Matrix dNorm2Out    = MatrixMultiplication.multiply(dFFN1Pre, W1);
        Matrix dRes1FromFFN = norm2.backward(dNorm2Out);

        // Residual 2: gradient zu res1 = direkt + FFN-Zweig
        Matrix dRes1 = gradOutput.add(dRes1FromFFN);

        // ── Attention rückwärts ───────────────────────────────────────────────
        Matrix dNorm1Out  = attention.backward(dRes1);
        Matrix dXFromAttn = norm1.backward(dNorm1Out);

        // Residual 1: direkt + Attention-Zweig
        return dRes1.add(dXFromAttn);
    }

    // ── Gradient-Management ───────────────────────────────────────────────────

    public void zeroGradients() {
        gradW1 = new Matrix(ffnDim, embDim);
        gradW2 = new Matrix(embDim, ffnDim);
        gradB1 = new double[ffnDim];
        gradB2 = new double[embDim];
        norm1.zeroGradients();
        norm2.zeroGradients();
        attention.zeroGradients();
    }

    // ── Adam ──────────────────────────────────────────────────────────────────

    /** Aktualisiert alle Parameter des Blocks via Adam. */
    public void adamStep(double lr, double beta1, double beta2, double eps, int t, int batchSize) {
        if (mW1 == null) {
            mW1 = new double[ffnDim][embDim]; vW1 = new double[ffnDim][embDim];
            mW2 = new double[embDim][ffnDim]; vW2 = new double[embDim][ffnDim];
            mB1 = new double[ffnDim];         vB1 = new double[ffnDim];
            mB2 = new double[embDim];         vB2 = new double[embDim];
        }
        double bc1 = 1 - Math.pow(beta1, t);
        double bc2 = 1 - Math.pow(beta2, t);

        SelfAttentionLayer.adamUpdate(W1.getValues(), gradW1.getValues(), mW1, vW1, lr, beta1, beta2, eps, bc1, bc2, batchSize);
        SelfAttentionLayer.adamUpdate(W2.getValues(), gradW2.getValues(), mW2, vW2, lr, beta1, beta2, eps, bc1, bc2, batchSize);
        adamUpdate1d(b1, gradB1, mB1, vB1, lr, beta1, beta2, eps, bc1, bc2, batchSize);
        adamUpdate1d(b2, gradB2, mB2, vB2, lr, beta1, beta2, eps, bc1, bc2, batchSize);

        norm1.adamStep(lr, beta1, beta2, eps, t, batchSize);
        norm2.adamStep(lr, beta1, beta2, eps, t, batchSize);
        attention.adamStep(lr, beta1, beta2, eps, t, batchSize);
    }

    static void adamUpdate1d(double[] w, double[] g, double[] m, double[] v,
                              double lr, double beta1, double beta2, double eps,
                              double bc1, double bc2, int batchSize) {
        for (int i = 0; i < w.length; i++) {
            double gi = g[i] / batchSize;
            m[i] = beta1 * m[i] + (1 - beta1) * gi;
            v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
            w[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
        }
    }

    // ── Hilfsmethoden ─────────────────────────────────────────────────────────

    private Matrix relu(Matrix m) {
        double[][] v   = m.getValues();
        double[][] out = new double[v.length][v[0].length];
        for (int i = 0; i < v.length; i++)
            for (int j = 0; j < v[0].length; j++)
                out[i][j] = Math.max(0.0, v[i][j]);
        return new Matrix(out);
    }

    private Matrix reluBackward(Matrix grad, Matrix pre) {
        double[][] gv  = grad.getValues();
        double[][] pv  = pre.getValues();
        double[][] out = new double[gv.length][gv[0].length];
        for (int i = 0; i < gv.length; i++)
            for (int j = 0; j < gv[0].length; j++)
                out[i][j] = pv[i][j] > 0.0 ? gv[i][j] : 0.0;
        return new Matrix(out);
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

    public LayerNorm          getNorm1()     { return norm1; }
    public LayerNorm          getNorm2()     { return norm2; }
    public MultiHeadAttention getAttention() { return attention; }
    public Matrix   getW1()     { return W1; }
    public Matrix   getW2()     { return W2; }
    public double[] getB1()     { return b1; }
    public double[] getB2()     { return b2; }
    public Matrix   getGradW1() { return gradW1; }
    public Matrix   getGradW2() { return gradW2; }
}
