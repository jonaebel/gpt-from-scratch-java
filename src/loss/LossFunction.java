package loss;

import math.Matrix;

public interface LossFunction {

    public double compute(Matrix predictions, Matrix targets);
    public Matrix gradient(Matrix predictions, Matrix targets);

    public LossFunctionType getType();
}
