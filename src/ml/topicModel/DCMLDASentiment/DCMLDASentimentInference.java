package ml.topicModel.DCMLDASentiment;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.preprocessing.DataSetGenerator;
import ml.topicModel.utils.QuickSort;


/**
 * @author wenzhe
 */
public class DCMLDASentimentInference {
    DataSet dataset;
    DCMLDASentimentModel model;
    Options option;
    
    public DCMLDASentimentInference(DataSet dataset){
        this.dataset = dataset;
    }
    
    public void initModel(Options options){
        model = new DCMLDASentimentModel();
        model.init(options, dataset);
        this.option = options;
    }
    

    public void runSampler() throws FileNotFoundException, UnsupportedEncodingException{
        // burn-in period
        System.out.println("start burn-in period.......");
        for (int itr = 0 ; itr < 300 ; itr++){
            System.out.println("gibbs sampling: " + itr + " iteration");
            model.runGibbsSampler();
            if (itr >1 && itr % 100 == 0){
                model.updateParamters();
                sentimentClassification(itr);
            }
            
        }
        System.out.println("finished burn-in period, move to sampling period.......");
        
        // sampling period
        for (int itr = 0 ; itr < 10 ; itr++){
            System.out.println("sampling period: " + itr);
            // first run gibbs sampler..
            for (int i = 0; i < 100; i++){
                System.out.println("gibbs sampling: " + i + " iteration");
                model.runGibbsSampler();  
            }
            // updating alpha and beta..
            System.out.println("updating hyperparameters");
            model.updateHyperparameters(itr+2);
            //printTopWords(itr+2);
            model.updateParamters();
            sentimentClassification(itr);
            //System.out.println("perplexity score is: " + model.getPerplexityScore());
        }
    }
    
    /**
     * Print the top words on the console, also write it into file. 
     */
    public void printTopWords(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("result/DCMLDA/" + "_topWords_"+itr+".txt", "UTF-8");
        System.out.println("Printing top words.....");
        writer.println("Printing top words....");
        
        // beta is the prior for DCMLDA, which plays similar role as \psi in LDA. 
        double[][][] beta = model.getBeta();
        
        int len1 = beta.length;
        int len2 = beta[0].length;
        double[][] temp = new double[len1][len2];
        for (int i = 0; i < len1; i++){
            for (int j =0; j < len2; j++){
                temp[i][j] = beta[i][j][0];
            }
        }
        
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        double[][] topWordsProbability = new double[option.K][tTop];
        
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            int vocabSize = dataset.vocab.getVocabularySize();
            System.out.println("Top words for topic: " + k);
            
            
            int[] index = new int[vocabSize];
            for (int v = 0; v < vocabSize; v++){
                index[v] = v;
            }
            
            QuickSort.quicksort(temp[k], index);
           
            for (int i = 0; i < tTop; i++){
                topWords[k][i] = dataset.vocab.indexTotokenMap.get(index[vocabSize-i-1]); 
                topWordsProbability[k][i] = beta[k][index[vocabSize-i-1]][0];
            }
        }
        
        for (int k = 0; k < option.K; k++){
            writer.println("Top words for topic: " + k);
            System.out.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i] + ":     " + topWordsProbability[k][i]);
                writer.println(topWords[k][i] + ":     " + topWordsProbability[k][i]);
            }
            
            System.out.println("****************************************");
            writer.println("****************************************");
        }
        
        writer.close();
    }
    
    /*
     * Sentiment classification, and write result into file and console. 
     */
    public void sentimentClassification(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        int positive = 0;
        int negative = 0;
        PrintWriter writer = new PrintWriter("result/DCMLDASentiment/"+dataset.dataSetName+"_sentimentClassification_"+itr+".txt", "UTF-8");
        writer.println("Doing sentiment classification");
        System.out.println("Doing sentiment classification.....");
        int total = 0;  // total number of documents have ratings between 0-2.5 or 3.5-5
        int correct = 0;
        float[][] pi = model.getPi();
        int predicted = 0;
        int actual = 0;
        for (int i = 0; i < dataset.getDocumentCount(); i++){
            double rating = dataset.getDocument(i).getRating();
            if (rating > 2 && rating < 4)
                continue;
            total++;
            if (pi[i][0] > pi[i][1])
                predicted = 0;
            else
                predicted = 1;
            
            if (rating <= 2){
                actual = 1;
                negative++;
            }
            else {
                actual = 0;
                positive++;
            }
            
            if (predicted == actual){
                correct++;
          
            }
            
        }
        double accuracy = correct*1.0/total;
        System.out.println("sentiment classification accuracy is: " + accuracy);
        writer.println("sentiment classification accuracy is: " + accuracy);
        writer.close();
        
        System.out.println("positive count:" + positive);
        System.out.println("negative count:" + negative);
    }
    
    
    public static void executeYelpDataSet() throws IOException{
        DataSet dataset = DataSetGenerator.createYelpDataSetForWordLevel("data/yelp");
        DCMLDASentimentInference inference = new DCMLDASentimentInference(dataset);
        Options opt = new Options();
       
        inference.initModel(opt);
        inference.runSampler();
    }
    
    public static void main (String[] args) throws IOException{
        DCMLDASentimentInference.executeYelpDataSet();
    }
}
