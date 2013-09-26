package ml.topicModel.ASUM;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ml.topicModel.ngram.QuickSort;

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
    
    public void runSampler(){
        int niter = option.niters;
        for (int itr = 0; itr < niter; itr++){
            System.out.println("gibbs sampling: " + itr + " iteration");
            model.runSampler();
        }
        
        model.updateParamters();
    }
    
public void printTopWordsPositive(){
        
        System.out.println("Top words for positive");
        
        double[][][] phi = model.getTopicWordDistribution();
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            int vocabSize = dataset.vocab.getVocabularySize();
            
            int[] index = new int[vocabSize];
            for (int v = 0; v < vocabSize; v++){
                index[v] = v;
            }
            QuickSort.quicksort(phi[0][k], index);
           
            for (int i = 0; i < tTop; i++){
                topWords[k][i] = dataset.vocab.indexTotokenMap.get(index[vocabSize-i-1]); 
            }
        }
        
        for (int k = 0; k < option.K; k++){
            System.out.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i]);
            }
            
            System.out.println("****************************************");
        }
    } 
    
    public void printTopWordsNegative(){
        System.out.println("Top words for negative");
    
        double[][][] phi = model.getTopicWordDistribution();
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            int vocabSize = dataset.vocab.getVocabularySize();
        
        
        
            int[] index = new int[vocabSize];
            for (int v = 0; v < vocabSize; v++){
                index[v] = v;
            }
            QuickSort.quicksort(phi[1][k], index);
       
            for (int i = 0; i < tTop; i++){
                topWords[k][i] = dataset.vocab.indexTotokenMap.get(index[vocabSize-i-1]); 
            }
        }
    
        for (int k = 0; k < option.K; k++){
            System.out.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i]);
            }
        
            System.out.println("****************************************");
        }
    } 

    public static void main(String[] args) throws IOException{
        
        DataSet dataset = new DataSet("data/yelp");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
        opt.alpha = 2;
        opt.beta = 0.01;
        opt.niters = 2000;
        opt.K = 20;
        opt.tWords = 20;
        
        inference.initModel(opt);
        inference.runSampler();
        
        inference.printTopWordsPositive();
        inference.printTopWordsNegative();
        
    }
}
