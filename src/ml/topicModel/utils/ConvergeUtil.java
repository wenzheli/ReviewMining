package ml.topicModel.utils;

/**
 * Check convergence of ARRAY or MATRIX. 
 * Author: Barbieri et al. "Probabilistic topic models for sequence data"
 */
public class ConvergeUtil {
    
    public static boolean matrixConverged(double[][][] a, double[][][] b,
            double threshold) {

        for (int i = 0; i < a.length; ++i)
            if (!matrixConverged(a[i], b[i], threshold))
                return false;

        return true;
    }
    
    public static boolean matrixConverged(double[][] a, double[][] b,
            double threshold) {

        for (int i = 0; i < a.length; ++i)
            if (!arrayConverged(a[i], b[i], threshold))
                return false;

        return true;
    }
    
    /**
     * 
     * @param a
     *            first array
     * @param b
     *            second array
     * @param threshold
     * @return true if |a - b|/|a| < threshold false otherwise
     */
    public static boolean arrayConverged(double a[], double b[],
            double threshold) {

        double a2sum = 0;
        double difference2 = 0;

        for (int i = 0; i < a.length; i++) {
            a2sum += a[i] * a[i];
            difference2 += (a[i] - b[i]) * (a[i] - b[i]);
        }
        /**
         * sqrt ( (a-b)^2/a^2 )< threshold
         */
        // System.out.println("A2sum::: "+a2sum);
        if (Math.sqrt(difference2 / a2sum) < threshold)
            return true;
        else
            return false;
    }// arrayConverged
}
