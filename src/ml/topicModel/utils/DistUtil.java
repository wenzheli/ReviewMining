package ml.topic.utils;

/**
 * Distribution-related utils. 
 * @author wenzhe
 *
 */
public class DistUtil {
    /**
     * sample the index based on the multinomial distribution. 
     * @return
     */
    public static int sampleFromMultinomial(double[] p){
        int n = p.length;
        for (int i = 1; i < n; i++)
            p[i] += p[i-1];
        double u = Math.random() * p[n-1];
        int idx;
        for (idx = 0; idx < n; idx++){
            if (u <= p[idx])
                break;
        }

        return idx;
    }
}
