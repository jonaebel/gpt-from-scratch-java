package initialization;

import math.Matrix;

public interface WeightInitialization {
        /*
          - fanIn —> the number of inputs a neuron receives (i.e., how many neurons feed into it)
          - fanOut —>i the number of outputs a neuron sends to (i.e., how many neurons it feeds into in the next layer)
     */

    Matrix compute(int fanIn, int fanOut);
    WeightInitializationTyp getTyp();

}
