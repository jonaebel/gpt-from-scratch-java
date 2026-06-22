package activation;

public enum ActivationFunctionType {
    SIGMOID("sigmoid"),
    RELU("relu"),
    LINEAR("linear"),
    SOFTMAX("softmax"),
    TANH("tanh"),
    LEAKY_RELU("leaky_relu");

    private final String id;

    ActivationFunctionType(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public static ActivationFunctionType fromId(String id) {
        for (ActivationFunctionType t : values()) {
            if (t.id.equals(id)) return t;
        }
        throw new IllegalArgumentException("Unknown activation function: " + id);
    }
}
