package loss;

import math.Matrix;

/*
    Categorical Cross-Entropy (CCE)
        Loss:     L = -(1/N) * sum_i sum_k ( y_ik * log(y_hat_ik) )
        Gradient: dL/dy_hat = -(1/N) * y / y_hat  (elementweise)
        Wird bei Mehrklassen-Klassifikation mit One-Hot-Targets verwendet.
        Setzt voraus, dass predictions Softmax-Ausgaben sind (Summe pro Zeile = 1).
        Predictions werden geclipt um log(0) zu vermeiden.
 */
public class CategoricalCrossEntropy implements LossFunction {

    private static final double EPS = 1e-12;

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pClipped = Math.max(EPS, p[i][j]);
                sum += t[i][j] * Math.log(pClipped);
            }
        return -sum / N;
    }

    @Override
    public Matrix gradient(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        int N = p.length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pClipped = Math.max(EPS, p[i][j]);
                grad[i][j] = -(1.0 / N) * t[i][j] / pClipped;
            }
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.CATEGORICAL_CROSS_ENTROPY;
    }
}
