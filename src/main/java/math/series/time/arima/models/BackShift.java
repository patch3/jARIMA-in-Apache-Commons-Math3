package math.series.time.arima.models;

import lombok.Getter;
import lombok.val;
import math.series.time.arima.core.ArimaException;

/**
 * Backshift operator for handling lags in ARIMA.
 * Implements polynomial operations on time series.
 */
public final class BackShift {
    @Getter
    private final int degree;  // maximum lag, e.g. AR(1) degree will be 1
    private final boolean[] indices;
    private int[] offsets = null;
    private double[] coeffs = null;

    //Constructor
    public BackShift(int degree, boolean initial) {
        if (degree < 0) {
            throw new ArimaException("degree must be non-negative");
        }
        this.degree = degree;
        this.indices = new boolean[this.degree + 1];
        for (var j = 0; j <= this.degree; ++j) {
            this.indices[j] = initial;
        }
        this.indices[0] = true; // zero index must be true all the time
    }

    public BackShift(boolean[] indices, boolean copyIndices) throws ArimaException {
        if (indices == null) {
            throw new ArimaException("null indices given");
        }
        this.degree = indices.length - 1;
        if (copyIndices) {
            this.indices = new boolean[degree + 1];
            System.arraycopy(indices, 0, this.indices, 0, degree + 1);
        } else {
            this.indices = indices;
        }
    }

    public double[] getCoefficientsFlattened() {
        if (degree <= 0 || offsets == null || coeffs == null) {
            return new double[0];
        }
        var temp = -1;
        for (var offset : offsets) {
            if (offset > temp) {
                temp = offset;
            }
        }
        val maxIdx = 1 + temp;
        val flattened = new double[maxIdx];
        for (var j = 0; j < offsets.length; ++j) {
            flattened[offsets[j]] = coeffs[j];
        }
        return flattened;
    }

    public void setIndex(int index, boolean enable) {
        indices[index] = enable;
    }

    /**
     * Applies another backshift operator, combining lags.
     *
     * @param another the other backshift operator
     * @return the new combined operator
     */
    public BackShift apply(BackShift another) {
        var mergedDegree = degree + another.degree;
        var merged = new boolean[mergedDegree + 1];
        for (var j = 0; j <= mergedDegree; ++j) {
            merged[j] = false;
        }
        for (var j = 0; j <= degree; ++j) {
            if (indices[j]) {
                for (var k = 0; k <= another.degree; ++k) {
                    merged[j + k] = merged[j + k] || another.indices[k];
                }
            }
        }
        return new BackShift(merged, false);
    }

    public void initializeParams(boolean includeZero) {
        indices[0] = includeZero;
        var nonzeroCount = 0;
        for (var j = 0; j <= degree; ++j) {
            if (indices[j]) {
                ++nonzeroCount;
            }
        }
        offsets = new int[nonzeroCount]; // cannot be 0 as 0-th index is always true
        coeffs = new double[nonzeroCount];
        var coeffIndex = 0;
        for (var j = 0; j <= degree; ++j) {
            if (indices[j]) {
                offsets[coeffIndex] = j;
                coeffs[coeffIndex] = 0;
                ++coeffIndex;
            }
        }
    }

    // MAKE SURE to initializeParams before calling below methods
    public int numParams() {
        return offsets.length;
    }

    public int[] paramOffsets() {
        return offsets;
    }

    public double getParam(final int paramIndex) throws ArimaException {
        for (var j = 0; j < offsets.length; ++j) {
            if (offsets[j] == paramIndex) {
                return coeffs[j];
            }
        }
        throw new ArimaException("invalid parameter index: " + paramIndex);
    }

    public double[] getAllParam() {
        return this.coeffs;
    }

    public void setParam(final int paramIndex, final double paramValue) throws ArimaException {
        var offsetIndex = -1;
        for (var j = 0; j < offsets.length; ++j) {
            if (offsets[j] == paramIndex) {
                offsetIndex = j;
                break;
            }
        }
        if (offsetIndex == -1) {
            throw new ArimaException("invalid parameter index: " + paramIndex);
        }
        coeffs[offsetIndex] = paramValue;
    }

    public double getLinearCombinationFrom(double[] timeseries, int tsOffset) {
        var linearSum = 0.0;
        for (var j = 0; j < offsets.length; ++j) {
            linearSum += timeseries[tsOffset - offsets[j]] * coeffs[j];
        }
        return linearSum;
    }
}
