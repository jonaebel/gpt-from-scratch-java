package initialization;

import math.Matrix;

import java.util.Random;

/*
    He Normal initialization: weights ~ N(0, sqrt(2 / fanIn))
    Designed for layers with ReLU activations.
 */
public class HeNormal implements WeightInitialization {

    private final Random random = new Random();

    @Override
    public Matrix compute(int fanIn, int fanOut) {
        double stddev = Math.sqrt(2.0 / fanIn);
        double[][] values = new double[fanIn][fanOut];
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                values[i][j] = random.nextGaussian() * stddev;
        return new Matrix(values);
    }

    @Override
    public WeightInitializationTyp getTyp() {
        return WeightInitializationTyp.HE_NORMAL;
    }
}
