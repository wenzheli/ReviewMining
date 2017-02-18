package ml.topicModel.NGramSentiment;


public class DistributionUtils {
    /**
     * sample the index based on the distributions.. 
     * @return
     */
    public static int getSample(double[] p){
        int n = p.length;
        for (int i = 1; i < n; i++)
            p[i] += p[i-1];
        double u = Math.random() * p[n-1];
        int idx;
        for (idx = 0; idx < n; idx++){
            if (u < p[idx])
                break;
        }

        return idx;
    }
    
   
    
    public static LatentVariable getSample(double[][][] p){
        int len1 = p.length;
        int len2 = p[0].length;
        int len3 = p[0][0].length;
        
        int n = len1 * len2 * len3;
        double[] temp = new double[n];
        int cnt = 0;
        for (int i = 0; i < len1; i++){
            for (int j = 0; j < len2; j++){
                for (int k = 0; k < len3; k++){
                    temp[cnt] = p[i][j][k];
                    cnt++;
                }
            }
        }
        
        for (int i = 1; i < n; i++){
            temp[i] += temp[i-1];
        }
           
        double u = Math.random() * temp[n-1];
        int idx;
        for (idx = 0; idx < n; idx++){
            if (u <= temp[idx])
                break;
        }
        int sentiment = idx / (len2 * len3);
        int remainder = idx % (len2 * len3);
        int topic = remainder / len3;
        int indicatorValue = remainder%len3;
      
     
        return new LatentVariable(sentiment, topic, indicatorValue);
    }
    
}
