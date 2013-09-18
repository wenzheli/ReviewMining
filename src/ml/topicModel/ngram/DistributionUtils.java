package ml.topicModel.ngram;

import ml.topicModel.lda.LatentVariable;

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
    
    public static LatentVariable getSample(double[][] p){
        int n = p.length * p[0].length;
        double[] temp = new double[n];
        int cnt = 0;
        for (int i = 0; i < p.length; i++){
            for (int j = 0; j < p[0].length; j++){
                temp[cnt] = p[i][j];
                cnt++;
            }
        }
        
        for (int i = 1; i < n; i++){
            temp[i] += temp[i-1];
        }
           
        double u = Math.random() * temp[n-1];
        int idx;
        for (idx = 0; idx < n; idx++){
            if (u < temp[idx])
                break;
        }
        int topic;
        int indicatorValue;
        
      
            topic = idx/2;
            indicatorValue = idx%2;
     
        return new LatentVariable(topic, indicatorValue);
    }
    
}
