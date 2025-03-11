package math.arima.models;

import lombok.Getter;
import math.arima.core.ArimaException;

/**
 * Helper class that implements polynomial of back-shift operator
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
            throw new RuntimeException("degree must be non-negative");
        }
        this.degree = degree;
        this.indices = new boolean[this.degree + 1];
        for (int j = 0; j <= this.degree; ++j) {
            this.indices[j] = initial;
        }
        this.indices[0] = true; // zero index must be true all the time
    }

    public BackShift(boolean[] indices, boolean copyIndices) {
        if (indices == null) {
            throw new RuntimeException("null indices given");
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
        int temp = -1;
        for (int offset : offsets) {
            if (offset > temp) {
                temp = offset;
            }
        }
        final int maxIdx = 1 + temp;
        final double[] flattened = new double[maxIdx];
        for (int j = 0; j < offsets.length; ++j) {
            flattened[offsets[j]] = coeffs[j];
        }
        return flattened;
    }

    public void setIndex(int index, boolean enable) {
        indices[index] = enable;
    }

    public BackShift apply(BackShift another) {
        int mergedDegree = degree + another.degree;
        boolean[] merged = new boolean[mergedDegree + 1];
        for (int j = 0; j <= mergedDegree; ++j) {
            merged[j] = false;
        }
        for (int j = 0; j <= degree; ++j) {
            if (indices[j]) {
                for (int k = 0; k <= another.degree; ++k) {
                    merged[j + k] = merged[j + k] || another.indices[k];
                }
            }
        }
        return new BackShift(merged, false);
    }

    public void initializeParams(boolean includeZero) {
        indices[0] = includeZero;
        int nonzeroCount = 0;
        for (int j = 0; j <= degree; ++j) {
            if (indices[j]) {
                ++nonzeroCount;
            }
        }
        offsets = new int[nonzeroCount]; // cannot be 0 as 0-th index is always true
        coeffs = new double[nonzeroCount];
        int coeffIndex = 0;
        for (int j = 0; j <= degree; ++j) {
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

    public double getParam(final int paramIndex) {
        for (int j = 0; j < offsets.length; ++j) {
            if (offsets[j] == paramIndex) {
                return coeffs[j];
            }
        }
        throw new ArimaException("invalid parameter index: " + paramIndex);
    }

    public double[] getAllParam() {
        return this.coeffs;
    }

    public void setParam(final int paramIndex, final double paramValue) {
        int offsetIndex = -1;
        for (int j = 0; j < offsets.length; ++j) {
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

    public void copyParamsToArray(double[] dest) {
        System.arraycopy(coeffs, 0, dest, 0, coeffs.length);
    }

    public double getLinearCombinationFrom(double[] timeseries, int tsOffset) {
        double linearSum = 0;
        for (int j = 0; j < offsets.length; ++j) {
            linearSum += timeseries[tsOffset - offsets[j]] * coeffs[j];
        }
        return linearSum;
    }


}
