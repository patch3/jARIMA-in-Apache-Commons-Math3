package math.arima.analytics;

import lombok.val;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Time series forecasting Utilities
 */
public final class ForecastUtil {
    public static final double testSetPercentage = 0.15;
    public static final double maxConditionNumber = 100;
    public static final double confidence_constant_95pct = 1.959963984540054;

    /**
     * Instantiates Toeplitz matrix from given input array
     *
     * @param input double array as input data
     * @return a Toeplitz InsightsMatrix
     */
    public static RealMatrix initToeplitz(double[] input) {
        val length = input.length;
        val toeplitz = new double[length][length];

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                if (j > i) {
                    toeplitz[i][j] = input[j - i];
                } else if (j == i) {
                    toeplitz[i][j] = input[0];
                } else {
                    toeplitz[i][j] = input[i - j];
                }
            }
        }
        return new Array2DRowRealMatrix(toeplitz);
    }

    /**
     * Invert AR part of ARMA to obtain corresponding MA series
     *
     * @param ar      AR portion of the ARMA
     * @param ma      MA portion of the ARMA
     * @param lag_max maximum lag
     * @return MA series
     */
    public static double[] ARMAtoMA(final double[] ar, final double[] ma, final int lag_max) {
        final int p = ar.length;
        final int q = ma.length;
        final double[] psi = new double[lag_max];

        for (int i = 0; i < lag_max; i++) {
            double tmp = (i < q) ? ma[i] : 0.0;
            for (int j = 0; j < Math.min(i + 1, p); j++) {
                tmp += ar[j] * ((i - j - 1 >= 0) ? psi[i - j - 1] : 1.0);
            }
            psi[i] = tmp;
        }
        final double[] include_psi1 = new double[lag_max];
        include_psi1[0] = 1;
        System.arraycopy(psi, 0, include_psi1, 1, lag_max - 1);
        return include_psi1;
    }

    /**
     * Simple helper that returns cumulative sum of coefficients
     *
     * @param coeffs array of coefficients
     * @return array of cumulative sum of the coefficients
     */
    public static double[] getCumulativeSumOfCoeff(final double[] coeffs) {
        final int len = coeffs.length;
        final double[] cumulativeSquaredCoeffSumVector = new double[len];
        double cumulative = 0.0;
        for (int i = 0; i < len; i++) {
            cumulative += Math.pow(coeffs[i], 2);
            cumulativeSquaredCoeffSumVector[i] = Math.pow(cumulative, 0.5);
        }
        return cumulativeSquaredCoeffSumVector;
    }
}
