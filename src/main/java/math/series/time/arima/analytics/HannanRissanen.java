package math.series.time.arima.analytics;

import lombok.val;
import math.series.time.arima.core.ArimaException;
import math.series.time.arima.models.ArimaParameterModel;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.util.Arrays;

/**
 * Implementation of the Hannan-Rissanen algorithm for ARMA parameter estimation.
 * Based on the paper by Hyndman and Khandakar (2008).
 */
public final class HannanRissanen {
    private static final double LAMBDA = 1e-6;

    /**
     * Estimates ARMA model parameters using an iterative refinement method.
     *
     * @param data_orig      the original time series data
     * @param params         the ARIMA model to populate with parameters
     * @param forecast_length the length of the forecast interval
     * @param maxIteration   the maximum number of iterations
     * @throws ArimaException if there is insufficient data for estimation
     */
    public static void estimateARMA(final double[] data_orig, final ArimaParameterModel params,
                                    final int forecast_length, final int maxIteration) {
        val data = Arrays.copyOf(data_orig, data_orig.length);
        val totalLength = data.length;
        val r = Math.max(params.getDegreeP(), params.getDegreeQ()) + 1;
        val length = totalLength - forecast_length;
        val size = length - r;

        if (length < 2 * r) {
            throw new ArimaException("Not enough data points: length=" + length + ", r=" + r);
        }

        val errors = new double[length];
        Arrays.fill(errors, 0, r, 0.0);

        // Исправлено: размерность матрицы [size][p+q]
        val matrix = new double[size][params.getNumParamsP() + params.getNumParamsQ()];

        var bestRMSE = -1.0;
        var remainIteration = maxIteration;
        double[] bestParams = null;
        while (--remainIteration >= 0) {
            val estimatedParams = iterationStep(params, data, errors, matrix, r, length, size);
            params.setParamsFromVector(estimatedParams);

            val forecasts = ArimaSolver.forecastARMA(params, data, length, data.length);
            val anotherRMSE = ArimaSolver.computeRMSE(
                    data,
                    forecasts,
                    length,
                    0,
                    forecast_length
            );

            updateErrors(data, errors, params, r, size);

            if (bestRMSE < 0 || anotherRMSE < bestRMSE) {
                bestParams = estimatedParams;
                bestRMSE = anotherRMSE;
            }
        }
        params.setParamsFromVector(bestParams);
    }

    private static double[] iterationStep(
            final ArimaParameterModel params,
            final double[] data, final double[] errors,
            final double[][] matrix, final int r, final int length, final int size) {

        fillMatrix(matrix, params, data, errors, r, size);
        val zt = new Array2DRowRealMatrix(matrix, false);
        val x = new ArrayRealVector(data, r, size);

        RealMatrix ztz = zt.transpose().multiply(zt);

        RealVector ztx = zt.transpose().operate(x);

        try {
            // Matrix regularization to ensure positive certainty
            final RealMatrix regularizedZtz = ztz.copy();
            for (int i = 0; i < regularizedZtz.getRowDimension(); i++) {
                regularizedZtz.addToEntry(i, i, LAMBDA);
            }
            return new CholeskyDecomposition(regularizedZtz).getSolver().solve(ztx).toArray();
        } catch (NonPositiveDefiniteMatrixException e) {
            return new LUDecomposition(ztz).getSolver().solve(ztx).toArray();
        }
    }

    private static void fillMatrix(double[][] matrix, ArimaParameterModel params,
                                   double[] data, double[] errors, int r, int size) {
        var colIdx = 0;
        // Filling in the AR part
        for (var pIdx : params.getOffsetsAR()) {
            for (var i = 0; i < size; i++) {
                matrix[i][colIdx] = data[r - pIdx + i];
            }
            colIdx++;
        }
        // Filling in the MA part
        for (var qIdx : params.getOffsetsMA()) {
            for (var i = 0; i < size; i++) {
                matrix[i][colIdx] = errors[r - qIdx + i];
            }
            colIdx++;
        }
    }

    private static void updateErrors(double[] data, double[] errors,
                                     ArimaParameterModel params, int r, int size) {
        val trainForecasts = new ArrayRealVector(
                ArimaSolver.forecastARMA(params, data, r, data.length)
        );
        for (var j = 0; j < size; ++j) {
            errors[j + r] = data[j + r] - trainForecasts.getEntry(j);
        }
    }

    public static RealVector fit(final double[] data, final int p) {
        val rVector = getVector(data, p);
        // Creating the Greenhouse Matrix
        val toeplitz = ForecastUtil.initToeplitz(rVector.getSubVector(0, p).toArray());
        // Solving a system with a check for degeneracy
        try {
            return new CholeskyDecomposition(toeplitz).getSolver().solve(rVector.getSubVector(1, p));
        } catch (NonPositiveDefiniteMatrixException e) {
            return new LUDecomposition(toeplitz).getSolver().solve(rVector.getSubVector(1, p));
        }
    }

    private static RealVector getVector(double[] data, int p) {
        val length = data.length;
        if (length == 0 || p < 1) {
            throw new RuntimeException("Invalid parameters: length=" + length + ", p=" + p);
        }
        // Correct calculation of autocovariance
        val dataMatrix = new Array2DRowRealMatrix(new double[][]{data});
        val covariance = new Covariance(dataMatrix);
        val covarianceMatrix = covariance.getCovarianceMatrix();

        val rVector = new ArrayRealVector(p + 1);
        for (var j = 0; j <= p; j++) {
            rVector.setEntry(j, covarianceMatrix.getEntry(0, j));
        }
        return rVector;
    }
}