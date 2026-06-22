package network;

import math.Matrix;

import java.util.Arrays;

/**
 * Layer Normalization normalisiert jeden Token-Vektor (Zeile) unabhängig:
 *
 *   x_hat  = (x − mean) / sqrt(var + eps)
 *   output = gamma * x_hat + beta
 *
 * Enthält eigenen Adam-Zustand, damit TransformerBlock.adamStep() direkt
 * delegieren kann, ohne externe Moment-Arrays zu verwalten.
 *
 * Gradient-Akkumulation: backward() addiert (+=) zu den Gradienten.
 * Vor jedem Batch muss zeroGradients() aufgerufen werden.
 */
public class LayerNorm {

    private final int    dim;
    private final double eps;

    double[] gamma;     // [dim]  Skalierung — init 1
    double[] beta;      // [dim]  Verschiebung — init 0

    double[] gradGamma; // [dim]
    double[] gradBeta;  // [dim]

    // Adam-Momente (lazy init beim ersten adamStep)
    private double[] mGamma, vGamma;
    private double[] mBeta,  vBeta;

    // Zwischengespeichert für backward()
    private Matrix   lastX;
    private Matrix   lastXNorm;
    private double[] lastVar;

    public LayerNorm(int dim) {
        this.dim = dim;
        this.eps = 1e-5;

        gamma     = new double[dim];
        beta      = new double[dim];
        gradGamma = new double[dim];
        gradBeta  = new double[dim];

        Arrays.fill(gamma, 1.0);
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /** X: [seqLen × dim] → [seqLen × dim] */
    public Matrix forward(Matrix X) {
        lastX = X;
        double[][] xv   = X.getValues();
        int rows = xv.length, cols = xv[0].length;

        double[][] xNorm = new double[rows][cols];
        double[][] out   = new double[rows][cols];
        lastVar = new double[rows];

        for (int i = 0; i < rows; i++) {
            double mean = 0;
            for (int j = 0; j < cols; j++) mean += xv[i][j];
            mean /= cols;

            double var = 0;
            for (int j = 0; j < cols; j++) { double d = xv[i][j] - mean; var += d * d; }
            var /= cols;
            lastVar[i] = var;

            double invStd = 1.0 / Math.sqrt(var + eps);
            for (int j = 0; j < cols; j++) {
                xNorm[i][j] = (xv[i][j] - mean) * invStd;
                out[i][j]   = gamma[j] * xNorm[i][j] + beta[j];
            }
        }
        lastXNorm = new Matrix(xNorm);
        return new Matrix(out);
    }

    // ── Backward (akkumulierend) ───────────────────────────────────────────────

    /**
     * Addiert Gradienten zu gradGamma/gradBeta (nicht überschreiben).
     * Vor dem ersten Batch muss zeroGradients() aufgerufen worden sein.
     *
     *   gradOutput: [seqLen × dim] → dX [seqLen × dim]
     */
    public Matrix backward(Matrix gradOutput) {
        double[][] dout  = gradOutput.getValues();
        double[][] xNorm = lastXNorm.getValues();
        int rows = dout.length, cols = dout[0].length;

        // gradGamma/gradBeta akkumulieren (kein Reset hier!)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                gradGamma[j] += dout[i][j] * xNorm[i][j];
                gradBeta[j]  += dout[i][j];
            }

        // dX via vollständige Ableitung durch Mittelwert + Varianz
        double[][] dx = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double invStd = 1.0 / Math.sqrt(lastVar[i] + eps);

            double[] dxHat = new double[cols];
            for (int j = 0; j < cols; j++) dxHat[j] = dout[i][j] * gamma[j];

            double sum1 = 0, sum2 = 0;
            for (int j = 0; j < cols; j++) { sum1 += dxHat[j]; sum2 += dxHat[j] * xNorm[i][j]; }

            for (int j = 0; j < cols; j++)
                dx[i][j] = invStd * (dxHat[j] - sum1 / cols - xNorm[i][j] * sum2 / cols);
        }
        return new Matrix(dx);
    }

    // ── Gradient-Management ───────────────────────────────────────────────────

    public void zeroGradients() {
        Arrays.fill(gradGamma, 0.0);
        Arrays.fill(gradBeta,  0.0);
    }

    // ── Adam ──────────────────────────────────────────────────────────────────

    /**
     * Aktualisiert gamma und beta via Adam.
     * @param batchSize Anzahl Samples — Gradienten werden dadurch normiert
     */
    public void adamStep(double lr, double beta1, double beta2, double eps, int t, int batchSize) {
        if (mGamma == null) {
            mGamma = new double[dim]; vGamma = new double[dim];
            mBeta  = new double[dim]; vBeta  = new double[dim];
        }
        double bc1 = 1 - Math.pow(beta1, t);
        double bc2 = 1 - Math.pow(beta2, t);

        for (int j = 0; j < dim; j++) {
            double gG = gradGamma[j] / batchSize;
            mGamma[j] = beta1 * mGamma[j] + (1 - beta1) * gG;
            vGamma[j] = beta2 * vGamma[j] + (1 - beta2) * gG * gG;
            gamma[j] -= lr * (mGamma[j] / bc1) / (Math.sqrt(vGamma[j] / bc2) + eps);

            double gB = gradBeta[j] / batchSize;
            mBeta[j]  = beta1 * mBeta[j]  + (1 - beta1) * gB;
            vBeta[j]  = beta2 * vBeta[j]  + (1 - beta2) * gB * gB;
            beta[j]  -= lr * (mBeta[j] / bc1) / (Math.sqrt(vBeta[j] / bc2) + eps);
        }
    }

    // ── Getter ────────────────────────────────────────────────────────────────

    public double[] getGamma()     { return gamma; }
    public double[] getBeta()      { return beta; }
    public double[] getGradGamma() { return gradGamma; }
    public double[] getGradBeta()  { return gradBeta; }
}
