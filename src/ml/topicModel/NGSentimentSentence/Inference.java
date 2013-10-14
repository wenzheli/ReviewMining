package ml.topicModel.NGSentimentSentence;

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
    NGramModelSentiSentence model;
    Options option;
    
    public Inference(DataSet dataset){
        this.dataset = dataset;
    }
    
    public void initModel(Options options){
        model = new NGramModelSentiSentence();
        model.init(options, dataset);
        this.option = options;
    }
    
    public void initDataSet(){
        
    }
    
    public void runSampler() throws FileNotFoundException, UnsupportedEncodingException{
        int niter = option.niters;
        for (int itr = 0; itr < niter; itr++){
            System.out.println("gibbs sampling: " + itr + " iteration");
            model.runSampler();
            
            if (itr % option.savestep == 0){
                model.updateParamters();
                //printTopWords(itr);
                sentimentClassification(itr);
            }
        }
        
        model.updateParamters();
    }
    
    /*
     * Sentiment classification, and write result into file and console. 
     */
    public void sentimentClassification(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        int positive = 0;
        int negative = 0;
        PrintWriter writer = new PrintWriter("result/NGramSentimentSentence/"+dataset.dataSetName+"_sentimentClassification_"+itr+".txt", "UTF-8");
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
      
        double[][] phi = model.getTopicWordDistribution();
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            Map<Double, Integer> map = new TreeMap<Double, Integer>();
            for (int v = 0; v < dataset.vocab.getVocabularySize(); v++){
                map.put(phi[k][v], v);
            }
            
            Collection<Integer> indices = map.values();
            Object[] objects =  indices.toArray();
            int[] sortedIndexs = new int[objects.length];
            for (int i = 0; i < objects.length; i++){
                sortedIndexs[i] = Integer.parseInt(objects[i].toString());
            }
            for (int i = 0; i < tTop; i++){
                topWords[k][i] = dataset.vocab.indexTotokenMap.get(sortedIndexs[objects.length-i-1]); 
            }
        }
        
        for (int k = 0; k < option.K; k++){
            System.out.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i]);
            }
            
            System.out.println("****************************************");
        }
        
        // also, write into file
        PrintWriter writer = new PrintWriter("result/yelp_unigram-"+itr+".txt", "UTF-8");
        for (int k = 0; k < option.K; k++){
            writer.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                writer.println(topWords[k][i]);
            }
            
            writer.println("****************************************");
        }
        writer.close();
    } 
    
    public void printTopWordsFromNGram(int itr) throws FileNotFoundException, UnsupportedEncodingException{
        double[][] phi = model.getTopWordsFromNGram();
        Map<Integer, List<Integer>> indexToNGramMap = model.getIndexToNGramMap();
        int vocabSize = indexToNGramMap.keySet().size();
        int tTop = option.tWords; // get the tTop words from each topic
        String[][] topWords = new String[option.K][tTop];
        for (int k = 0; k < option.K; k++){
            // select the top words for topic k
            int[] index = new int[vocabSize];
            for (int v = 0; v < vocabSize; v++){
                index[v] = v;
            }
            QuickSort.quicksort(phi[k], index);
           
            for (int i = 0; i < tTop; i++){
                List<Integer> termIndexs = indexToNGramMap.get(index[vocabSize-i-1]);
                topWords[k][i] = convertIndexToWords(termIndexs);
            }
        }
        
        for (int k = 0; k < option.K; k++){
            System.out.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                System.out.println(topWords[k][i]);
            }
            
            System.out.println("****************************************");
        }
        
        
        // also, write into file
        PrintWriter writer = new PrintWriter("result/yelp_ngram-"+itr+".txt", "UTF-8");
        for (int k = 0; k < option.K; k++){
            writer.println("Top words for topic: " + k);
            for (int i = 0; i < topWords[k].length; i++){
                writer.println(topWords[k][i]);
            }
            
            writer.println("****************************************");
        }
        writer.close();
    }
    
    private String convertIndexToWords(List<Integer> indexs){
        String nGram= "";
        for (Integer idx : indexs){
            nGram += dataset.vocab.indexTotokenMap.get(idx) + " ";
        }
        
        return nGram;
    }
    
    public static void executeYelpDataSet() throws IOException{
        DataSet dataset = DataSetGenerator.createYelpDataSetForSentenceLevel("data/yelp");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
       
        inference.initModel(opt);
        inference.runSampler();
    }
    
    public static void main(String[] args) throws IOException{
        executeYelpDataSet();
    }
}
