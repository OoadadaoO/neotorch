package app.adada.neo4j.util;

import java.util.List;

public class TypeParser {
    public static <T> List<T> parseList(Object value, Class<T> type, List<T> defaultValue) {
        if (value instanceof List<?> list) {
            return list.stream()
                    .filter(type::isInstance)
                    .map(type::cast)
                    .toList();
        }
        return defaultValue;
    }

    public static <T> List<T> parseList(Object value, Class<T> type) {
        return parseList(value, type, List.of());
    }

    public static <T> T parse(Object value, Class<T> type, T defaultValue) {
        if (value == null) {
            return defaultValue;
        }
        if (type.isInstance(value)) {
            return type.cast(value);
        }
        throw new IllegalArgumentException("Cannot cast " + value + " to " + type.getName());
    }

    public static <T> T parse(Object value, Class<T> type) {
        return parse(value, type, null);
    }
}
