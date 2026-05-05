package com.bot.recorder.recordWriter;

import java.util.List;

public class JsonLineBuilder {
    private final StringBuilder out;
    private boolean first = true;
    private boolean finished = false;

    public static class RawJson {
        private final String json;

        public RawJson(String json) {
            this.json = json;
        }
    }

    public JsonLineBuilder(int initialCapacity) {
        this.out = new StringBuilder(initialCapacity);
        this.out.append('{');
    }


    private void appendJsonArray(StringBuilder out, List<?> values) {
        out.append('[');

        for (int i = 0; i < values.size(); i++) {
            if (i > 0) out.append(',');
            appendJsonValue(out, values.get(i));
        }

        out.append(']');
    }

    private void appendJsonValue(StringBuilder out, Object value) {
        if (value == null) {
            out.append("null");
        } else if (value instanceof String) {
            appendJsonString(out, (String) value);
        } else if (value instanceof Number) {
            appendJsonNumber(out, (Number) value);
        } else if (value instanceof Boolean) {
            out.append(((Boolean) value).booleanValue() ? "true" : "false");
        } else if (value instanceof List) {
            appendJsonArray(out, (List<?>) value);
        } else if (value instanceof RawJson) {
            out.append(((RawJson) value).json);
        } else {
            appendJsonString(out, value.toString());
        }
    }

    private void appendJsonNumber(StringBuilder out, Number number) {
        if (number instanceof Float) {
            float value = ((Float) number).floatValue();
            if (Float.isNaN(value) || Float.isInfinite(value)) {
                out.append("null");
                return;
            }
        } else if (number instanceof Double) {
            double value = ((Double) number).doubleValue();
            if (Double.isNaN(value) || Double.isInfinite(value)) {
                out.append("null");
                return;
            }
        }

        out.append(number.toString());
    }

    private void appendJsonString(StringBuilder out, String value) {
        out.append('"');

        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
                    break;
                case '\b':
                    out.append("\\b");
                    break;
                case '\f':
                    out.append("\\f");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        out.append("\\u");
                        String hex = Integer.toHexString(c);
                        for (int j = hex.length(); j < 4; j++) out.append('0');
                        out.append(hex);
                    } else {
                        out.append(c);
                    }
                    break;
            }
        }

        out.append('"');
    }


    public JsonLineBuilder reset() {
        out.setLength(0);
        out.append('{');
        first = true;
        finished = false;
        return this;
    }

    public JsonLineBuilder field(String name, Object value) {
        if (!first) out.append(',');
        first = false;
        appendJsonString(out, name);
        out.append(':');
        appendJsonValue(out, value);
        return this;
    }

    public String finish() {
        if (!finished) {
            out.append('}');
            finished = true;
        }

        return out.toString();
    }
}

