package loss;

import math.Matrix;

/*
    Mean Squared Error (MSE)
        Loss:     L = (1/N) * sum( (y_hat - y)^2 )
        Gradient: dL/dy_hat = (2/N) * (y_hat - y)
        Typisch fuer Regressionsprobleme. Bestraft grosse Abweichungen
        staerker als kleine (quadratische Strafe).
 */
public class MSE implements LossFunction {

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length * p[0].length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double diff = p[i][j] - t[i][j];
                sum += diff * diff;
            }
        return sum / N;
    }

    @Override
    public Matrix gradient(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        int N = p.length * p[0].length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++)
                grad[i][j] = (2.0 / N) * (p[i][j] - t[i][j]);
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.MSE;
    }
}
