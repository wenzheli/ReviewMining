package ml.topic.utils;

public class FunctionUtil {
    /**
     * Estimates the Digamma value of a real positive number.
     * 
     * @param x
     *            Input of the Digamma function.
     * @return The Digamma value for <code>x</code>.
     */
    private  static double digamma(double x) {
        double p;
        assert x > 0;
        x = x + 6;
        p = 1 / (x * x);
        p = (((0.004166666666667 * p - 0.003968253986254) * p + 0.008333333333333)
                * p - 0.083333333333333)
                * p;
        p = p + Math.log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3)
                - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6);
        return p;
    }// digamma
    
    
    /**
     * private static double digamma(double x) { double pow = 1; double ret =
     * Math.log(x) - .5 * x;
     * 
     * for (int i = 0, n = BERNOULLI_NUMBERS.length; i < n; ++i) { pow *= x * x;
     * ret -= BERNOULLI_NUMBERS[i] / pow; }
     * 
     * return ret; }
     */
    private static double logGamma(double x) {
        double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
        double ser = 1.0 + 76.18009173    / (x + 0)   - 86.50532033    / (x + 1)
                         + 24.01409822    / (x + 2)   -  1.231739516   / (x + 3)
                         +  0.00120858003 / (x + 4)   -  0.00000536382 / (x + 5);
        return tmp + Math.log(ser * Math.sqrt(2 * Math.PI));
     }
}
