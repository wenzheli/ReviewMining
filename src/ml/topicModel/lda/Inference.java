package ml.topicModel.LDA;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import ml.topicModel.LDA.Options;
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
        int saveStep = option.savestep;
        int niter = option.niters;
        for (int itr = 0; itr < niter; itr++){
            System.out.println("gibbs sampling: " + itr + " iteration");
            model.runSampler();
            
            if (itr % saveStep == 0){
                model.updateParamters();
                printTopWords(itr);
                System.out.println("perplexity score is: " + model.getPerplexityScore());
            }
        }
        
        model.updateParamters();
    }
    
    
    public void printTopWords(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("result/LDA/"+dataset.dataSetName + "_topWords_"+itr+".txt", "UTF-8");
        System.out.println("Printing top words");
        writer.println("Printing top words");
        double[][] phi = model.getTopicWordDistribution();
        
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
                topWordsProbability[k][i] = phi[k][index[vocabSize-i-1]];
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
    
    public static void executeYelpDataSet() throws IOException{
        DataSet dataset = DataSetGenerator.createYelpDataSetForWordLevel("data/yelp");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
       
        inference.initModel(opt);
        inference.runSampler();
    }
    
    public static void executeNIPSDataSet() throws IOException{
        DataSet dataset = DataSetGenerator.createNIPSDataSetForWordLevel("data/nipstxt");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
       
        inference.initModel(opt);
        inference.runSampler();
    }
    
    public static void main(String[] args) throws IOException{
        Inference.executeNIPSDataSet();
    }

}
