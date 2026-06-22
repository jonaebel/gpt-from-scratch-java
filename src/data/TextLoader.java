package data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class TextLoader {

    // Liest die Datei und gibt den gesamten Inhalt als String zurück
    public static String load(String path) throws IOException {
        return new String(Files.readAllBytes(Paths.get(path)));
    }

}
