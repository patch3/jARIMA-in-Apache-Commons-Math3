package math.series.time.arima.analytics;

import lombok.val;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Utilities for time series forecasting.
 * Contains methods for working with matrices and ARMA transformations.
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
        val n = input.length;
        val matrix = new Array2DRowRealMatrix(n, n);

        for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) {
                matrix.setEntry(i, j, input[Math.abs(i - j)]);
            }
        }
        return matrix;
    }

    /**
     * Converts ARMA parameters to an MA representation.
     *
     * @param ar      the AR coefficients
     * @param ma      the MA coefficients
     * @param lag_max the maximum lag for calculation
     * @return an array of MA coefficients
     */
    public static double[] ARMAtoMA(final double[] ar, final double[] ma, final int lag_max) {
        val p = ar.length;
        val q = ma.length;
        val psi = new double[lag_max];

        for (var i = 0; i < lag_max; i++) {
            var tmp = (i < q) ? ma[i] : 0.0;
            for (var j = 0; j < Math.min(i + 1, p); j++) {
                tmp += ar[j] * ((i - j - 1 >= 0) ? psi[i - j - 1] : 1.0);
            }
            psi[i] = tmp;
        }
        val include_psi1 = new double[lag_max];
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
        val len = coeffs.length;
        val cumulativeSquaredCoeffSumVector = new double[len];
        var cumulative = 0.0;
        for (var i = 0; i < len; i++) {
            cumulative += Math.pow(coeffs[i], 2);
            cumulativeSquaredCoeffSumVector[i] = Math.pow(cumulative, 0.5);
        }

        return cumulativeSquaredCoeffSumVector;
    }
}
