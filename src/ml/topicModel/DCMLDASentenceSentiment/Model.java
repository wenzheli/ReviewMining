package ml.topicModel.DCMLDASentenceSentiment;


import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ml.topicModel.common.data.LatentVariable;
import ml.topicModel.common.data.SDocument;
import ml.topicModel.common.data.Sentence;
import ml.topicModel.common.data.Vocabulary;
import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.data.Document;
import ml.topicModel.utils.ConvergeUtil;
import ml.topicModel.utils.DistUtil;
import ml.topicModel.utils.DistributionUtils;
import ml.topicModel.utils.SparseMatrix;

/**
 * DCMLDA model is extension of LDA model, which can capture the burstiness of the
 * topic. For instance, the word that already appears in certain document, is likely to 
 * be appeared again in this document. 
 * 
 * For each document in LDA, we draw words from global topic-word distribution
 * \phi, but in DCMLDA, we draw words from document specific topic word distribution.
 * 
 * The model implements the collapsed gibbs sampler for inference, which is very similar
 * to LDA model, except that we draw word from document-specific distributions. However,
 * the most big difference in terms of inference is that we need to learn the hyperparameters
 * for DCMLDA model explicitly. The topic-word distribution prior \beta in DCMLDA actually
 * plays the similar role as \phi (global topic word distribution) in LDA. 
 * 
 * In order to learn these priors, we are using fixed point iteration method. Basically, 
 * we first calculate the likelihood for all the data given priors, and try to 
 * maximize that function using iterative manner. For more information about inference
 * part, please look at the report that comes with this code. 
 * 
 * @author wenzhe   nadalwz1115@hotmail.com
 *
 */
public class Model {
    // max iteration needed for updating alpha priors. 
    public final static int MAX_ALPHA_ITERATIONS = 30;
    // max iteration needed for updating beta priors
    public final static int MAX_BETA_ITERATIONS = 30;
    public final static int MAX_GAMMA_ITERATIONS = 30;
    
    private int D;      // number of documents in the data set
    private int V;      // vocabulary size
    private int K;      // number of topics
    private int S;      // number of sentiments. 
    
    /**
     * Dirichlet hyperparameters. In DCMLDA model, these hyperparameters are very 
     * important. Thus, we need to estimate those explicitly. 
     */
    private double[] alpha;   // size K
    private double alphaSum;  // size 1
    private double[] gamma;   // size S
    private double gammaSum;  // size 1
    private double[][][] beta;  // size S * K * V
    private double[][] betaSum; // size S * K
    
    /**
     * Parameters we need to estimate from the data. 
     */
    private float [][] theta;   // document-topic distributions, size D x K
    private float [][][] phi;   // doc-topic-word distributions, size D x K x V 
                                // in DCMLDA model, we keep each topic-word distribution for each document
    private float [][] pi;      // document-sentiment distribution, size D * S 
    private int[][] z; // latent variable, topic assignments for each sentence.
    private int[][] l; // latent variable, sentiment assignment for each sentence
    
    /**
     * Counting variables. Those variables are sufficient statistics for latent variable z and l
     */
    private SparseMatrix [][] wDocSentimentTopicWord; // wDocSentimentTopicWord[i][s][k][v]: number of token v, that are in document i, and assigned to sentiment s,topic k, size D*S*K*V
    private int [][][] wDocSentimentTopic;            // wDocSentimentTopic[i][s][k]: # of words in document i, assigned to sentiment s, and topic k. 
    private int [] sDoc;                              // sDoc[i]: total number of sentences in document i, size D
    private int [][] sDocSentiment;                   // sDocSentiment[i][s]: number of sentences in document i, assigned to sentiment s
    private int [][][] sDocSentimentTopic;            // sDocSentimentTopic[i][s][k]: number of sentences in document i that assigned to sentiment s and topic k, size D x S x K
    private DataSet dataset;                          // input data set for learning.
     
    // initialize parameters. 
    public void init(Options options, DataSet dataset){
        S = options.S;
        K = options.K;
        this.dataset = dataset;
        Vocabulary vocab = dataset.getVocabulary();
        D = dataset.getDocumentCount();
        V = dataset.getVocabulary().getVocabularySize();
        pi = new float[D][S];
        // initialize hyperparameters using default values. 
        alpha = new double[K];
        beta = new double[S][K][V];
        betaSum = new double[S][K];
        gamma = new double[S];
        // initialize beta
        for (int s = 0; s < S; s++){
            for (int k = 0; k < K; k++){
                for (int v = 0; v < V; v++){
                    beta[s][k][v] = 0.001;
                }
            }
        }
        
        Set<Integer> positiveWords = vocab.getPositiveWordS();
        for (Integer tokenId : positiveWords){
            for (int s = 0; s <S; s++){
                for (int k = 0; k < K; k++){
                    beta[1][k][tokenId] = 0;
                }
            }
          
        }
        
        Set<Integer> negativeWords = vocab.getNegativeWords();
        for (Integer tokenId : negativeWords){
            for (int s = 0; s < S; s++){
                for (int k = 0; k < K; k++){
                    beta[0][k][tokenId] = 0;
                }
            }
        }
        
        for (int s = 0; s< S; s++){
            for (int k = 0; k < K; k++){
                for (int v = 0; v < V; v++){
                    betaSum[s][k] += beta[s][k][v];
                }
            }
        }
        
       
        
        for (int k=0; k < K; k++){
            alpha[k] = options.alpha;
        }
        for (int s = 0; s < S; s++){
            gamma[s] = 1;
            gammaSum += gamma[s];
        }
        
        alphaSum = options.alpha * K;
        
        
        // allocate memory for counting variables. 
        wDocSentimentTopicWord = new SparseMatrix[D][S];
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                wDocSentimentTopicWord[i][s] = new SparseMatrix(K);
            }
        }
        wDocSentimentTopic = new int[D][S][K];
        sDoc = new int[D];
        sDocSentiment = new int[D][S];
        sDocSentimentTopic = new int[D][S][K];
        
        // initialize latent variable and counting variables for sentences.  
        z = new int[D][];
        l = new int[D][];
        for (int i = 0; i < D; i++){
            SDocument d = (SDocument)dataset.getDocument(i);
            int numSentences = d.getNumOfSentences();
            z[i] = new int[numSentences];
            l[i] = new int[numSentences];
            for (int j = 0; j < numSentences; j++){
                int randTopic = (int)(Math.random() * K);
                int randSentiment = (int)(Math.random() * S);
                z[i][j] = randTopic; 
                Sentence sentence = d.getSentence(j); 
                for (Integer tokenIdx : sentence.getTokens()){
                   if (vocab.positiveWords.contains(tokenIdx)){
                       randSentiment = 0;
                   } else if(vocab.negativeWords.contains(tokenIdx)){
                       randSentiment  = 1;
                   }
                }
                l[i][j] = randSentiment;
                                       
                for (Integer tokenIdx : sentence.getTokens()){
                    wDocSentimentTopicWord[i][l[i][j]].increment(randTopic, tokenIdx);
                    wDocSentimentTopic[i][l[i][j]][randTopic]++;
                }
                sDoc[i]++;
                sDocSentiment[i][randSentiment]++;
                sDocSentimentTopic[i][randSentiment][randTopic]++;  
            }  
        }   
    }
    
    // this will run one iteration of collapsed gibbs sampling.
    public void runGibbsSampler(){
        for (int i = 0; i < D; i++){
            SDocument d = (SDocument)dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfSentences(); j++){
                LatentVariable latentVariable = sample(i,j);
                z[i][j] = latentVariable.getTopic();
                l[i][j] = latentVariable.getSentiment();
            }
        }
    }
    
    /**
     * Sample topic for jth word in ith document. 
     */
    private LatentVariable sample(int i,  int j){
        SDocument d = (SDocument)dataset.getDocument(i);
        Sentence sentence = d.getSentence(j);
        // remove current word..
        int oldTopic = z[i][j];
        int oldSentiment = l[i][j];
        
        sDoc[i]--;
        sDocSentiment[i][oldSentiment]--;
        sDocSentimentTopic[i][oldSentiment][oldTopic]--;   
            
        for (Integer tokenIdx : sentence.getTokens()){
            wDocSentimentTopicWord[i][oldSentiment].decrement(oldTopic, tokenIdx);
            wDocSentimentTopic[i][oldSentiment][oldTopic]--;
        }
        
        
        Map<Integer, Integer> termCountMap = new HashMap<Integer, Integer>();
        for (Integer tokenIdx : sentence.getTokens()){
            if (!termCountMap.containsKey(tokenIdx)){
                termCountMap.put(tokenIdx, 1);
            } else{
                int cnt = termCountMap.get(tokenIdx);
                termCountMap.put(tokenIdx, cnt + 1);
            }
        }
        
        // compute p(z[i][j]|*)
        double[][] p = new double[S][K];
        for (int s = 0; s < S; s++){
            for (int k = 0; k < K; k++){
                    
                double devidend[] = new double[sentence.getTokens().size()];
                for (int itr = 0; itr < sentence.getTokens().size(); itr++){
                    devidend[itr] = (betaSum[s][k] + wDocSentimentTopic[i][s][k] + itr);
                }
                
                double term = 1;
                int count = 0;
                for (Integer tokenIdx : termCountMap.keySet()){
                    int cnt = termCountMap.get(tokenIdx);
                    for (int itr = 0; itr < cnt; itr++){
                        term = term *  (beta[s][k][tokenIdx] + wDocSentimentTopicWord[i][s].get(k, tokenIdx) + itr)/(devidend[count++]);
                    }
                }
                  
                p[s][k] = ((alpha[k] + sDocSentimentTopic[i][s][k])/(alphaSum + sDocSentiment[i][s])) 
                        * ((gamma[s]+sDocSentiment[i][s])/(gammaSum + sDoc[i]))
                        * term;
            }
        }
        
        // sample the topic topic from the distribution p[j].
        LatentVariable latentVariable = DistributionUtils.getSample(p);
        int newTopic = latentVariable.getTopic();
        int newSentiment = latentVariable.getSentiment();
        
        sDoc[i]++;
        sDocSentiment[i][newSentiment]++;
        sDocSentimentTopic[i][newSentiment][newTopic]++;      
        for (Integer tokenIdx : sentence.getTokens()){
            wDocSentimentTopicWord[i][newSentiment].increment(newTopic, tokenIdx);
            wDocSentimentTopic[i][newSentiment][newTopic]++;
        }
        
        return latentVariable;
    }
    
    public void updateHyperparameters(int iteration){
        updateAlpha(iteration);
        updateGamma(iteration);
        updateBeta(iteration);
    }
    
    /**
     * update alpha until it is converged or exceed the maximum iterations. 
     * @param iteration     the current iteration
     */
    protected void updateAlpha(int iteration) {
        int currIteration = 0;
        double[] previousAlpha = new double[K];
        boolean b;
        do {
            System.arraycopy(alpha, 0, previousAlpha, 0, K);
            updateAlpha0(iteration);
            currIteration++;
            b = ConvergeUtil.arrayConverged(alpha, previousAlpha, Math.pow(10, -6));
        } while (!b && currIteration < MAX_ALPHA_ITERATIONS);
    }
    
    protected void updateGamma(int iteration){
        int currIteration = 0;
        double[] previousGamma = new double[S];
        boolean b;
        do {
            System.arraycopy(gamma, 0, previousGamma, 0, S);
            updateGamma0(iteration);
            currIteration++;
            b = ConvergeUtil.arrayConverged(gamma, previousGamma, Math.pow(10, -6));
        } while (!b && currIteration < MAX_GAMMA_ITERATIONS);
    }
    
    /**
     * Update the beta Dirichlet parameter, topic-word distributions, until
     * it is converged or iteration exceed the maximum allowed one. 
     */
    protected void updateBeta(int iteration) {
        int currIteration = 0;
        double[][][] previousBeta = new double[S][K][V];

        boolean b;
        do {
            for (short s = 0; s < S; s++){
                for (short k = 0; k < K; ++k) {
                    System.arraycopy(beta[s][k], 0, previousBeta[s][k], 0, V);
                    updateBeta0(s,k, iteration);
                }
            }
            currIteration++;
            b = ConvergeUtil.matrixConverged(beta, previousBeta, Math.pow(10, -8));
        } while (!b && currIteration < MAX_BETA_ITERATIONS);
    }
    
    private void updateGamma0(int iteration){
        double dividend = 0;
        for (int i = 0; i < D; i++){
                for (int j = 0; j < sDoc[i]; j++){
                    dividend += 1.0/(j + gammaSum);
                }
        }
        
        for (int s = 0; s < S; s++){
            double divisor = 0;
            
            for (int i = 0; i < D; i++){   
                    for (int j = 0; j < sDocSentiment[i][s]; j++){
                        divisor += 1.0/(j + gamma[s]);
                    }
            }
            
            double newGamma = Math.exp(Math.log(gamma[s]) + Math.log(divisor)
                    - Math.log(dividend));
            
            if (Double.isNaN(newGamma) || Double.isInfinite(newGamma)
                    || newGamma < 0)
                newGamma = 0;

            newGamma = (gamma[s] * (iteration - 1) + newGamma)
                    / iteration;

            gammaSum -= gamma[s];
            gamma[s] = newGamma;
            gammaSum += gamma[s];
        }
    }
    
    /**
     * update alpha vector, using fixed point iteration. 
     * Please see report for reference.
     * @param iteration
     */
    private void updateAlpha0(int iteration){
        double dividend = 0;
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                for (int j = 0; j < sDocSentiment[i][s]; j++){
                    dividend += 1.0/(j + alphaSum);
                }
            }
            
        }
        
        for (int k = 0; k < K; k++){
            double divisor = 0;
            
            for (int i = 0; i < D; i++){
                for (int s = 0; s < S; s++){
                    for (int j = 0; j < sDocSentimentTopic[i][s][k]; j++){
                        divisor += 1.0/(j + alpha[k]);
                    }
                }
                
            }
            
            double newAlpha = Math.exp(Math.log(alpha[k]) + Math.log(divisor)
                    - Math.log(dividend));
            
            if (Double.isNaN(newAlpha) || Double.isInfinite(newAlpha)
                    || newAlpha < 0)
                newAlpha = 0;

            newAlpha = (alpha[k] * (iteration - 1) + newAlpha)
                    / iteration;

            alphaSum -= alpha[k];
            alpha[k] = newAlpha;
            alphaSum += alpha[k];
        }
    }
    
    
    /**
     * Update beta vector using fixed point iteration. For derivation, refer 
     * to the report. 
     * @param k              the kth beta vector
     * @param iteration      the current iteration
     */
    protected void updateBeta0(short s, short k, int iteration) {
        double dividend = 0;
        double divisor = 0; 
        
        for (int i = 0; i < D; i++)
            for (int j = 0; j < wDocSentimentTopic[i][s][k]; j++)
                dividend += 1.0 / (j + betaSum[s][k]);
        
        for (int v = 0; v < V; v++){
            divisor = 0;
            for (int i = 0; i < D; i++){
                for (int j = 0; j < wDocSentimentTopicWord[i][s].get(k, v); j++){
                    divisor += 1.0/(j + beta[s][k][v]);
                }
            }
            
            double newBeta_k = Math.exp(Math.log(beta[s][k][v]) + Math.log(divisor)
                    - Math.log(dividend));

            if (Double.isNaN(newBeta_k) || Double.isInfinite(newBeta_k)
                    || newBeta_k < 0)
                newBeta_k = 0;

            newBeta_k = (beta[s][k][v] * (iteration - 1) + newBeta_k)
                    / iteration;

            betaSum[s][k] -= beta[s][k][v];
            beta[s][k][v] = newBeta_k;
            betaSum[s][k] += beta[s][k][v];
        }
    }
    
    public void updateParamters(){
        // update pi
        for (int i = 0; i < D; i++){
            for (int s = 0; s < S; s++){
                pi[i][s] = (float)((sDocSentiment[i][s]+gamma[s])/(sDoc[i]+gammaSum));
            }
        }
        // update theta
        //for (int i = 0; i < D; i++){
        //    for (int k = 0; k < K; k++){
        //        theta[i][k] = (float) ((alpha[k] + nDocTopic[i][k]) / (alphaSum + nSentencesInDoc[i]));
        //    }
        //}
        
        // update phi
        //for (int i = 0; i < D; i++){
        //    for (int k = 0; k < K; k++){
        //        for (int v = 0; v < V; v++){
        //            phi[i][k][v] = (float) ((beta[k][v] + nDocTopicWord[i][k][v]) / (betaSum[k] + nDocTopic[i][k])); 
        //        }
        //    }
        //}
        
      
        
    }
    
    /**
     * Calculate the perplexity score for training set. 
     * @return
     */
    public double getPerplexityScore(){
        double ppx = 0;
        int size = 0;
        for (int i = 0; i < D; i++){
            Document d = dataset.getDocument(i);
            for (int j = 0; j < d.getNumOfTokens(); j++){
                int token = d.getToken(j);
                double prob = 0;
                for (int k = 0; k < K; k++){
                    prob += phi[i][k][token] * theta[i][k];
                }
                prob = Math.log(prob);
                if (!Double.isInfinite(prob) && !Double.isNaN(prob)){
                    ppx += prob;
                    size++;
                }
            }
        }      
        return Math.exp(-ppx/size);
    }
    
    public float[][] getPi(){
        return pi;
    }
    public float[][] getTopicDistribution(){
        return theta;
    }
    
    public float[][][] getDocTopicWordDistribution(){
        return phi;
    }
    
    public double[] getAlpha(){
        return alpha;
    }
    
    public double[][] getBeta(){
        return beta;
    }
}
