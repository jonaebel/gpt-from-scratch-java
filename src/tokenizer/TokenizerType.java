package tokenizer;

import loss.LossFunctionType;

public enum TokenizerType {
    CHARACKTERTOKENIZER("charackterTokenizer");

    private final String id;

    TokenizerType(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public static TokenizerType fromId(String id) {
        for (TokenizerType t : values()) {
            if (t.id.equals(id)) return t;
        }
        throw new IllegalArgumentException("Unknown loss function: " + id);
    }

}
