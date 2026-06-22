package activation;

import math.Matrix;

public class Tanh implements ActivationFunction {

    @Override
    public Matrix apply(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++)
                out[i][j] = Math.tanh(in[i][j]);
        return new Matrix(out);
    }

    @Override
    public Matrix derivative(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++)
            for (int j = 0; j < in[0].length; j++) {
                double t = Math.tanh(in[i][j]);
                out[i][j] = 1 - t * t;
            }
        return new Matrix(out);
    }

    @Override
    public ActivationFunctionType getType() {
        return ActivationFunctionType.TANH;
    }
}
