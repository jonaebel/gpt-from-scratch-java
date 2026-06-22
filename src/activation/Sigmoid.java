package activation;

import math.Matrix;

public class Sigmoid implements ActivationFunction {

    private double applyScalar(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public Matrix apply(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[i][j] = applyScalar(in[i][j]);
        return new Matrix(out);
    }

    @Override
    public Matrix derivative(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++) {
                double s = applyScalar(in[i][j]);
                out[i][j] = s * (1.0 - s);
            }
        return new Matrix(out);
    }

    @Override
    public ActivationFunctionType getType() {
        return ActivationFunctionType.SIGMOID;
    }
}
