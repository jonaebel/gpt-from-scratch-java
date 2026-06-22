package loss;

import math.Matrix;

/*
    Binary Cross-Entropy (BCE)
        Loss:     L = -(1/N) * sum( y*log(y_hat) + (1-y)*log(1-y_hat) )
        Gradient: dL/dy_hat = (1/N) * ( -y/y_hat + (1-y)/(1-y_hat) )
        Wird bei binaeren Klassifikationsproblemen verwendet (y in {0,1}).
        Predictions werden geclipt um log(0) zu vermeiden.
 */
public class BinaryCrossEntropy implements LossFunction {

    private static final double EPS = 1e-12;

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length * p[0].length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pClipped = Math.max(EPS, Math.min(1 - EPS, p[i][j]));
                sum += t[i][j] * Math.log(pClipped) + (1 - t[i][j]) * Math.log(1 - pClipped);
            }
        return -sum / N;
    }

    @Override
    public Matrix gradient(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        int N = p.length * p[0].length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pClipped = Math.max(EPS, Math.min(1 - EPS, p[i][j]));
                grad[i][j] = (1.0 / N) * (-t[i][j] / pClipped + (1 - t[i][j]) / (1 - pClipped));
            }
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.BINARY_CROSS_ENTROPY;
    }
}
