package math.arima.analytics;

import math.arima.models.ArimaParameterModel;
import org.apache.commons.math3.linear.*;

import java.util.Arrays;

public final class HannanRissanen {
    public static void estimateARMA(final double[] data_orig, final ArimaParameterModel params,
                                    final int forecast_length, final int maxIteration) {
        final double[] data = Arrays.copyOf(data_orig, data_orig.length);
        final int total_length = data.length;
        final int r = Math.max(params.getDegreeP(), params.getDegreeQ()) + 1;
        final int length = total_length - forecast_length;
        final int size = length - r;

        if (length < 2 * r) {
            throw new RuntimeException("Not enough data points: length=" + length + ", r=" + r);
        }

        final double[] errors = new double[length];
        Arrays.fill(errors, 0, r, 0.0);

        // Исправлено: размерность матрицы [size][p+q]
        final double[][] matrix = new double[size][params.getNumParamsP() + params.getNumParamsQ()];

        double bestRMSE = -1;
        int remainIteration = maxIteration;
        double[] bestParams = null;

        while (--remainIteration >= 0) {
            final double[] estimatedParams = iterationStep(params, data, errors, matrix, r, length, size);
            params.setParamsFromVector(estimatedParams);

            final double[] forecasts = ArimaSolver.forecastARMA(params, data, length, data.length);
            final double anotherRMSE = ArimaSolver.computeRMSE(
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
        final RealMatrix zt = new Array2DRowRealMatrix(matrix, false);
        final RealVector x = new ArrayRealVector(data, r, size);

        final RealMatrix ztz = zt.transpose().multiply(zt);

        final RealVector ztx = zt.transpose().operate(x);

        try {
            // Регуляризация матрицы для обеспечения положительной определённости
            final double lambda = 1e-6;
            final RealMatrix regularizedZtz = ztz.copy();
            for (int i = 0; i < regularizedZtz.getRowDimension(); i++) {
                regularizedZtz.addToEntry(i, i, lambda);
            }

            return new CholeskyDecomposition(regularizedZtz).getSolver().solve(ztx).toArray();

        } catch (NonPositiveDefiniteMatrixException e) {
            // Использование LU-декомпозиции как резервного метода
            return new LUDecomposition(ztz).getSolver().solve(ztx).toArray();
        }
    }

    private static void fillMatrix(double[][] matrix, ArimaParameterModel params,
                                   double[] data, double[] errors, int r, int size) {
        int colIdx = 0;
        // Заполнение AR-части
        for (int pIdx : params.getOffsetsAR()) {
            for (int i = 0; i < size; i++) {
                matrix[i][colIdx] = data[r - pIdx + i];
            }
            colIdx++;
        }
        // Заполнение MA-части
        for (int qIdx : params.getOffsetsMA()) {
            for (int i = 0; i < size; i++) {
                matrix[i][colIdx] = errors[r - qIdx + i];
            }
            colIdx++;
        }
    }

    private static void updateErrors(double[] data, double[] errors,
                                     ArimaParameterModel params, int r, int size) {
        final RealVector trainForecasts = new ArrayRealVector(
                ArimaSolver.forecastARMA(params, data, r, data.length)
        );
        for (int j = 0; j < size; ++j) {
            errors[j + r] = data[j + r] - trainForecasts.getEntry(j);
        }
    }

    public static RealVector fit(final double[] data, final int p) {
        final int length = data.length;
        if (length == 0 || p < 1) {
            throw new RuntimeException("Invalid parameters: length=" + length + ", p=" + p);
        }

        final RealVector rVector = computeAutocovariance(data, p);
        final RealMatrix toeplitz = ForecastUtil.initToeplitz(rVector.getSubVector(0, p).toArray());
        return solveToeplitzSystem(toeplitz, rVector.getSubVector(1, p));
    }

    private static RealVector computeAutocovariance(final double[] data, final int p) {
        final RealVector r = new ArrayRealVector(p + 1);
        r.setEntry(0, new ArrayRealVector(data).dotProduct(new ArrayRealVector(data)) / data.length);

        for (int j = 1; j <= p; j++) {
            double sum = 0;
            for (int i = 0; i < data.length - j; i++) {
                sum += data[i] * data[i + j];
            }
            r.setEntry(j, sum / data.length);
        }
        return r;
    }

    private static RealVector solveToeplitzSystem(final RealMatrix toeplitz, final RealVector rVector) {
        try {
            return new CholeskyDecomposition(toeplitz).getSolver().solve(rVector);
        } catch (SingularMatrixException e) {
            throw new RuntimeException("Toeplitz matrix is singular", e);
        }
    }
}