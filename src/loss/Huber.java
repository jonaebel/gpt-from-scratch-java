package loss;

import math.Matrix;

/*
    Huber Loss
        Sei r = y_hat - y (Residuum), delta der Schwellenwert.
        Loss pro Element:
            |r| <= delta:  0.5 * r^2          (quadratisch, wie MSE)
            |r| >  delta:  delta*(|r| - 0.5*delta)  (linear, wie MAE)
        Gradient pro Element:
            |r| <= delta:  r
            |r| >  delta:  delta * sign(r)
        Kombiniert die Robustheit des MAE gegenueber Ausreissern
        mit der Glattheit des MSE in der Naehe von 0.
 */
public class Huber implements LossFunction {

    private final double delta;

    public Huber(double delta) {
        this.delta = delta;
    }

    public Huber() {
        this(1.0);
    }

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length * p[0].length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double r = Math.abs(p[i][j] - t[i][j]);
                if (r <= delta)
                    sum += 0.5 * r * r;
                else
                    sum += delta * (r - 0.5 * delta);
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
            for (int j = 0; j < p[0].length; j++) {
                double r = p[i][j] - t[i][j];
                if (Math.abs(r) <= delta)
                    grad[i][j] = r / N;
                else
                    grad[i][j] = delta * Math.signum(r) / N;
            }
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.HUBER;
    }
}
