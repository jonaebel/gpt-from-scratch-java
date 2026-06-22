package loss;

public enum LossFunctionType {
    MSE("mse"),
    BINARY_CROSS_ENTROPY("binary_cross_entropy"),
    CATEGORICAL_CROSS_ENTROPY("categorical_cross_entropy"),
    HUBER("huber"),
    FOCAL_LOSS("focal_loss"),
    SOFTMAX_CCE("softmax_cce");

    private final String id;

    LossFunctionType(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public static LossFunctionType fromId(String id) {
        for (LossFunctionType t : values()) {
            if (t.id.equals(id)) return t;
        }
        throw new IllegalArgumentException("Unknown loss function: " + id);
    }
}
