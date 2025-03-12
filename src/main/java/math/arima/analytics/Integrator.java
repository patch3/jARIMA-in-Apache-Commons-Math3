package math.arima.analytics;

import lombok.val;
import math.arima.core.ArimaException;

/**
 * Helper class for differentiation and integration of time series.
 * Contains methods for transformations required in ARIMA modeling.
 */
public class Integrator {
    /**
     * General verification of input parameters for differentiation/integration.
     */
    private static void validateInputs(boolean isDifferentiate,
                                       double[] src, double[] dst,
                                       double[] initial, int d) {
        if (initial == null || initial.length != d || d == 0) {
            throw new ArimaException("Invalid initial: size=" + (initial != null ? initial.length : 0) + ", d=" + d);
        }
        if (isDifferentiate) {
            if (src == null || src.length <= d) {
                throw new ArimaException("Insufficient source size: " + (src != null ? src.length : 0) + ", d=" + d);
            }
            if (dst == null || dst.length != src.length - d) {
                throw new ArimaException("Invalid destination size: " + (dst != null ? dst.length : 0) + ", src=" + src.length + ", d=" + d);
            }
        } else {
            if (dst == null || dst.length <= d) {
                throw new ArimaException("Insufficient destination size: " + (dst != null ? dst.length : 0) + ", d=" + d);
            }
            if (src == null || src.length != dst.length - d) {
                throw new ArimaException("Invalid source size: " + (src != null ? src.length : 0) + ", dst=" + dst.length + ", d=" + d);
            }
        }
    }

    /**
     * Copying the initial conditions.
     */
    private static void copyInitialConditions(double[] src, double[] dst,
                                              double[] initial, int d,
                                              boolean isDifferentiate) {
        if (isDifferentiate) {
            System.arraycopy(src, 0, initial, 0, d);
        } else {
            System.arraycopy(initial, 0, dst, 0, d);
        }
    }

    /**
     * Performs differentiation of a time series.
     *
     * @param src     the source array of data
     * @param dst     the array to store the differentiated data
     * @param initial the initial conditions (values before differentiation)
     * @param d       the order of differentiation
     * @throws ArimaException if the parameters are invalid
     */
    public static void differentiate(final double[] src, final double[] dst,
                                     final double[] initial, final int d) {
        validateInputs(true, src, dst, initial, d);
        copyInitialConditions(src, dst, initial, d, true);

        for (int j = d, k = 0; j < src.length; ++j, ++k) {
            dst[k] = src[j] - src[k]; // src[k] для несезонного дифференцирования
        }
    }

    /**
     * Performs integration of a time series (inverse of differentiation).
     *
     * @param src     the source array of data (after differentiation)
     * @param dst     the array to store the restored data
     * @param initial the initial conditions (values before differentiation)
     * @param d       the order of integration
     * @throws ArimaException if the parameters are invalid
     */
    public static void integrate(final double[] src, final double[] dst,
                                 final double[] initial, final int d) {
        validateInputs(false, src, dst, initial, d);
        copyInitialConditions(src, dst, initial, d, false);

        for (int j = d, k = 0; k < src.length; ++j, ++k) {
            dst[j] = dst[k] + src[k]; // Накопление результата
        }
    }

    /**
     * Shifting the input data
     *
     * @param inputData   MODIFIED. input data
     * @param shiftAmount shift amount
     */
    public static void shift(double[] inputData, final double shiftAmount) {
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] += shiftAmount;
        }
    }

    /**
     * Compute the mean of input data
     *
     * @param data input data
     * @return mean
     */
    public static double computeMean(final double[] data) {
        if (data.length == 0) {
            return 0.0;
        }
        var sum = 0.0;
        for (double datum : data) {
            sum += datum;
        }
        return sum / data.length;
    }

    /**
     * Compute the variance of input data
     *
     * @param data input data
     * @return variance
     */
    public static double computeVariance(final double[] data) {
        val mean = computeMean(data);
        var variance = 0.0;
        for (double datum : data) {
            final double diff = datum - mean;
            variance += diff * diff;
        }
        return variance / (data.length - 1.0);
    }
}
