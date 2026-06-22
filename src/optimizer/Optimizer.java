package optimizer;

import network.Layer;

public interface Optimizer {

    void step(Layer[] layers);

    OptimizerType getTyp();

}
