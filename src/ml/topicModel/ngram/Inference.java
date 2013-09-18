package ml.topicModel.ngram;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Inference {
    DataSet dataset;
    NGramModel model;
    Options option;
    
    public Inference(DataSet dataset){
        this.dataset = dataset;
    }
    
    public void initModel(Options options){
        model = new NGramModel();
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
            
            if (itr % 50== 0){
                // we save the results for every 50 iterations...
                model.updateParamters();
                printTopWordsFromNGram(itr);
                printTopWords(itr);
            }
        }
        
        model.updateParamters();
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
    
    public static void main(String[] args) throws IOException{
        DataSet dataset = new DataSet("data/yelp");
        Inference inference = new Inference(dataset);
        Options opt = new Options();
        opt.alpha = 5;
        opt.beta = 0.01;
        opt.gamma = 0.1;
        opt.delta = 0.01;
        opt.niters = 10000;
        opt.K = 10;
        opt.tWords = 20;
        
        inference.initModel(opt);
        inference.runSampler();
        
        inference.printTopWords(opt.niters);  
        inference.printTopWordsFromNGram(opt.niters);
    }
}
