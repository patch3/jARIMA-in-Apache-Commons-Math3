package math.series.time.arima.core;

public class ArimaException extends RuntimeException {
    public ArimaException(String message) {
        super(message);
    }

    public ArimaException(String message, Throwable cause) {
        super(message, cause);
    }
}
