package loss;

import math.Matrix;

/*
    Focal Loss (Lin et al., 2017)
        Loss:     L = -(1/N) * sum( alpha * (1 - y_hat)^gamma * y*log(y_hat)
                                  + (1-alpha) * y_hat^gamma * (1-y)*log(1-y_hat) )
        Gradient: dL/dy_hat = (1/N) * alpha * (1-y_hat)^gamma * (-y/y_hat + gamma*y*log(y_hat)/(1-y_hat))
                            + (1/N) * (1-alpha) * y_hat^gamma * ((1-y)/(1-y_hat) - gamma*(1-y)*log(1-y_hat)/y_hat)
        Erweiterung der BCE fuer stark unausgewogene Datensaetze.
        Der Fokus-Term (1-p_t)^gamma reduziert den Verlust einfacher Beispiele,
        sodass das Training sich auf schwer zu klassifizierende Faelle konzentriert.
        alpha gewichtet die Klassen, gamma (>= 0) steuert die Fokussierung.
 */
public class FocalLoss implements LossFunction {

    private static final double EPS = 1e-12;

    private final double alpha;
    private final double gamma;

    public FocalLoss(double alpha, double gamma) {
        this.alpha = alpha;
        this.gamma = gamma;
    }

    public FocalLoss() {
        this(0.25, 2.0);
    }

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length * p[0].length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pC = Math.max(EPS, Math.min(1 - EPS, p[i][j]));
                double posLoss = alpha * Math.pow(1 - pC, gamma) * t[i][j] * Math.log(pC);
                double negLoss = (1 - alpha) * Math.pow(pC, gamma) * (1 - t[i][j]) * Math.log(1 - pC);
                sum += posLoss + negLoss;
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
                double pC = Math.max(EPS, Math.min(1 - EPS, p[i][j]));
                double posGrad = alpha * (Math.pow(1 - pC, gamma) * (-t[i][j] / pC)
                        + gamma * Math.pow(1 - pC, gamma - 1) * t[i][j] * Math.log(pC));
                double negGrad = (1 - alpha) * (Math.pow(pC, gamma) * (-(1 - t[i][j]) / (1 - pC))
                        + gamma * Math.pow(pC, gamma - 1) * (1 - t[i][j]) * Math.log(1 - pC));
                grad[i][j] = -(posGrad + negGrad) / N;
            }
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.FOCAL_LOSS;
    }
}
