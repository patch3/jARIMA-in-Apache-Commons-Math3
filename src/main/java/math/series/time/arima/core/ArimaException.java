package math.series.time.arima.core;

import math.series.time.SeriesException;

public class ArimaException extends SeriesException {
    public ArimaException(String message) {
        super(message);
    }

    public ArimaException(String message, Throwable cause) {
        super(message, cause);
    }
}
