package activation;

import math.Matrix;

public interface ActivationFunction {

    public Matrix apply(Matrix m);
    public Matrix derivative(Matrix m);

    ActivationFunctionType getType();

}
