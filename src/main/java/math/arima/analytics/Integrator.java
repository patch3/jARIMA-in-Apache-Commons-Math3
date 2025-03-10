package math.arima.analytics;

/**
 * Pure Helper Class
 */
public class Integrator {
    /**
     * Общая проверка входных параметров для дифференцирования/интегрирования.
     */
    private static void validateInputs(boolean isDifferentiate,
                                       double[] src, double[] dst,
                                       double[] initial, int d) {
        if (initial == null || initial.length != d || d == 0) {
            throw new RuntimeException("Invalid initial: size=" + (initial != null ? initial.length : 0) + ", d=" + d);
        }
        if (isDifferentiate) {
            if (src == null || src.length <= d) {
                throw new RuntimeException("Insufficient source size: " + (src != null ? src.length : 0) + ", d=" + d);
            }
            if (dst == null || dst.length != src.length - d) {
                throw new RuntimeException("Invalid destination size: " + (dst != null ? dst.length : 0) + ", src=" + src.length + ", d=" + d);
            }
        } else {
            if (dst == null || dst.length <= d) {
                throw new RuntimeException("Insufficient destination size: " + (dst != null ? dst.length : 0) + ", d=" + d);
            }
            if (src == null || src.length != dst.length - d) {
                throw new RuntimeException("Invalid source size: " + (src != null ? src.length : 0) + ", dst=" + dst.length + ", d=" + d);
            }
        }
    }

    /**
     * Копирование начальных условий.
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
     * Дифференцирование.
     *
     * @param src     source array of data
     * @param dst     destination array to store data
     * @param initial initial conditions
     * @param d       length of initial conditions
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
     * Интегрирование.
     *
     * @param src     source array of data
     * @param dst     destination array to store data
     * @param initial initial conditions
     * @param d       length of initial conditions
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
        final int length = data.length;
        if (length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double datum : data) {
            sum += datum;
        }
        return sum / length;
    }

    /**
     * Compute the variance of input data
     *
     * @param data input data
     * @return variance
     */
    public static double computeVariance(final double[] data) {
        double variance = 0.0;
        double mean = computeMean(data);
        for (double datum : data) {
            final double diff = datum - mean;
            variance += diff * diff;
        }
        return variance / (double) (data.length - 1);
    }
}
