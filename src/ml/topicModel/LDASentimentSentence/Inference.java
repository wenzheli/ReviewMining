package ml.topicModel.LDASentimentSentence;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.preprocessing.DataSetGenerator;
import ml.topicModel.utils.QuickSort;

public class Inference {
    DataSet dataset;
    LDAModel model;
    Options option;
    
    public Inference(DataSet dataset){
        this.dataset = dataset;
    }
    
    public void initModel(Options options){
        model = new LDAModel();
        model.init(options, dataset);
        this.option = options;
    }
    
    public void initDataSet(){
        
    }
    
    public void runSampler() throws FileNotFoundException, UnsupportedEncodingException{
        int niter = option.niters;
        int saveStep = option.saveStep;
        for (int itr = 0; itr < niter; itr++){
            System.out.println("gibbs sampling: " + itr + " iteration");
            model.runSampler();
            
            if (itr % saveStep == 0){
                model.updateParamters();
                printTopWords(itr);
                sentimentClassification(itr);
            }
        }   
    }
    
    /*
     * Sentiment classification, and write result into file and console. 
     */
    public void sentimentClassification(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        int positive = 0;
        int negative = 0;
        PrintWriter writer = new PrintWriter("result/LDASentimentSentence/"+dataset.dataSetName+"_sentimentClassification_"+itr+".txt", "UTF-8");
        writer.println("Doing sentiment classification");
        System.out.println("Doing sentiment classification.....");
        int total = 0;  // total number of documents have ratings between 0-2.5 or 3.5-5
        int correct = 0;
        double[][] pi = model.getSentimentDistribution();
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
    
    
    public void printTopWords(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("result/LDASentimentSentence/"+dataset.dataSetName+"_topWords_"+itr+".txt", "UTF-8");
        double[][][] phi = model.getTopicWordDistribution();
        
        // print top words for positive sentiment
        System.out.println("Top words for positive");
        writer.println("Top words for positive");
        printTopWords(phi[0], writer);
        
        // print top words for negative sentiment. 
        System.out.println("Top words for negative");
        writer.println("Top words for negative");
        printTopWords(phi[1], writer);
        
        writer.close();
    }
    
    private void printTopWords(double[][] phi, PrintWriter writer){
        int len1 = phi.length;
        int len2 = phi[0].length;
        double[][] temp = new double[len1][len2];
        for (int i = 0; i < len1; i++){
            for (int j =0; j < len2; j++){
                temp[i][j] = phi[i][j];
            }
        }
        
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        double[][] topWordsProbablity = new double[option.K][tTop];
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            int vocabSize = dataset.vocab.getVocabularySize();
            
            int[] index = new int[vocabSize];
            for (int v = 0; v < vocabSize; v++){
                index[v] = v;
            }
            QuickSort.quicksort(temp[k], index);
           
            for (int i = 0; i < tTop; i++){
                topWords[k][i] = dataset.vocab.indexTotokenMap.get(index[vocabSize-i-1]); 
                topWordsProbablity[k][i] = phi[k][index[vocabSize-i-1]];
            }
        }
        
        for (int k = 0; k < option.K; k++){
            System.out.println("Top words for topic: " + k);
            writer.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i] + ":     " + topWordsProbablity[k][i]);
                writer.println(topWords[k][i] + ":     " + topWordsProbablity[k][i]);
            }
            System.out.println("****************************************");
            writer.println("****************************************");
        }
    }
    
    
    public static void executeYelpDataSet() throws IOException{
        DataSet dataset = DataSetGenerator.createYelpDataSetForSentenceLevel("data/yelp");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
       
        inference.initModel(opt);
        inference.runSampler();
    }
    
    public static void executeNIPSDataSet() throws IOException{
        DataSet dataset = DataSetGenerator.createNIPSDataSetForSentenceLevel("data/nipstxt");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
       
        inference.initModel(opt);
        inference.runSampler();
    }
    
    public static void main(String[] args) throws IOException{
        Inference.executeYelpDataSet();
    }
}
