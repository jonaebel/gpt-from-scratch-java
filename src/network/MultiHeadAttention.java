package network;

import math.Matrix;
import math.MatrixMultiplication;

import java.util.Random;

/**
 * Multi-Head Attention — h parallele Attention-Köpfe, kombiniert via W_O.
 *
 *   head_i = SelfAttention_i(X)           [seqLen × dK]
 *   concat = Concat(head_0, …, head_{h-1}) [seqLen × embDim]
 *   output = concat @ W_O^T               [seqLen × embDim]
 *
 * Gradient-Akkumulation: backward() addiert zu gradW_O.
 * Vor jedem Batch: zeroGradients(). Nach jedem Batch: adamStep().
 */
public class MultiHeadAttention {

    private final int embDim;
    private final int numHeads;
    private final int dK;

    final SelfAttentionLayer[] heads;

    // Ausgabe-Projektion [embDim × embDim] — transponiert gespeichert
    Matrix W_O;
    Matrix gradW_O;

    // Adam-Momente für W_O (lazy init)
    private double[][] mW_O, vW_O;

    // Zwischengespeichert für backward()
    private Matrix lastConcat;

    public MultiHeadAttention(int embDim, int numHeads) {
        if (embDim % numHeads != 0)
            throw new IllegalArgumentException(
                "embDim (" + embDim + ") muss durch numHeads (" + numHeads + ") teilbar sein");

        this.embDim   = embDim;
        this.numHeads = numHeads;
        this.dK       = embDim / numHeads;

        heads = new SelfAttentionLayer[numHeads];
        for (int h = 0; h < numHeads; h++)
            heads[h] = new SelfAttentionLayer(embDim, dK);

        W_O    = randomMatrix(embDim, embDim, Math.sqrt(2.0 / (2 * embDim)));
        gradW_O = new Matrix(embDim, embDim);
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /** X: [seqLen × embDim] → [seqLen × embDim] */
    public Matrix forward(Matrix X) {
        int seqLen = X.getRows();

        Matrix[] headOutputs = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++)
            headOutputs[h] = heads[h].forward(X);

        lastConcat = concatenateColumns(headOutputs, seqLen);
        return MatrixMultiplication.multiplyBT(lastConcat, W_O);
    }

    // ── Backward (akkumulierend) ───────────────────────────────────────────────

    /** gradOutput: [seqLen × embDim] → dX [seqLen × embDim] */
    public Matrix backward(Matrix gradOutput) {
        // gradW_O akkumulieren
        gradW_O = gradW_O.add(MatrixMultiplication.multiplyATB(gradOutput, lastConcat));

        // dConcat = gradOutput @ W_O
        Matrix dConcat = MatrixMultiplication.multiply(gradOutput, W_O);

        // Auf Köpfe aufteilen und rückwärts
        Matrix[] dHeads = splitColumns(dConcat);
        Matrix dX = heads[0].backward(dHeads[0]);
        for (int h = 1; h < numHeads; h++)
            dX = dX.add(heads[h].backward(dHeads[h]));

        return dX;
    }

    // ── Gradient-Management ───────────────────────────────────────────────────

    public void zeroGradients() {
        gradW_O = new Matrix(embDim, embDim);
        for (SelfAttentionLayer head : heads) head.zeroGradients();
    }

    // ── Adam ──────────────────────────────────────────────────────────────────

    public void adamStep(double lr, double beta1, double beta2, double eps, int t, int batchSize) {
        if (mW_O == null) {
            mW_O = new double[embDim][embDim];
            vW_O = new double[embDim][embDim];
        }
        double bc1 = 1 - Math.pow(beta1, t);
        double bc2 = 1 - Math.pow(beta2, t);

        SelfAttentionLayer.adamUpdate(W_O.getValues(), gradW_O.getValues(),
                                      mW_O, vW_O, lr, beta1, beta2, eps, bc1, bc2, batchSize);

        for (SelfAttentionLayer head : heads)
            head.adamStep(lr, beta1, beta2, eps, t, batchSize);
    }

    // ── Hilfsmethoden ─────────────────────────────────────────────────────────

    private Matrix concatenateColumns(Matrix[] matrices, int seqLen) {
        double[][] result = new double[seqLen][embDim];
        for (int h = 0; h < numHeads; h++) {
            double[][] hv = matrices[h].getValues();
            for (int i = 0; i < seqLen; i++)
                System.arraycopy(hv[i], 0, result[i], h * dK, dK);
        }
        return new Matrix(result);
    }

    private Matrix[] splitColumns(Matrix m) {
        double[][] mv  = m.getValues();
        int seqLen = mv.length;
        Matrix[] result = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            double[][] part = new double[seqLen][dK];
            for (int i = 0; i < seqLen; i++)
                System.arraycopy(mv[i], h * dK, part[i], 0, dK);
            result[h] = new Matrix(part);
        }
        return result;
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

    public SelfAttentionLayer[] getHeads() { return heads; }
    public Matrix getW_O()                 { return W_O; }
    public Matrix getGradW_O()             { return gradW_O; }
    public void   setW_O(Matrix m)         { W_O = m; }
}
