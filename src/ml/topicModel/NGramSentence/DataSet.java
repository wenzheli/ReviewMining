package ml.topicModel.NGramSentence;

import ml.topicModel.common.preprocessing.PorterStemmer;
import ml.topicModel.common.preprocessing.StopWords;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class DataSet{
   
    List<Document> documents;
    int numDocuments;
    public Vocabulary vocab;
   
    public List<Document> getDocuments(){
        return documents;
    }
    
    public int getDocumentCount(){
        return documents.size();
    }
    
    public Vocabulary getVocabulary(){
        return vocab;
    }
    
    public Document getDocument(int index){
        return documents.get(index);
    }
    
    // for review data sets
    public DataSet(String filePath) throws IOException{
        BufferedReader br = null;
        String sCurrentLine = "";
        File file = new File(filePath);
        File[] files = file.listFiles();
        
        // we will count the number of tokens appears. This will be used for further filtering the words
        Map<String, Integer> tokenMap = new HashMap<String, Integer>();
        // first iteration, creating vocabulary list. 
        int fileCount = 0;
        for (File f: files){
            System.out.println("Process " + ++fileCount + "th file");
            int count = 0;
            br = new BufferedReader(new FileReader(f));
            while ((sCurrentLine = br.readLine()) != null){
                if (++count == 6){   
                    String review = sCurrentLine;
                    // remove the stop words
                    String[] tokens = review.split("\\s");
                    for (String token : tokens){
                        PorterStemmer stem = new PorterStemmer();
                        String stemmedToken = stem.stemming(token);
                       
                        
                        // remove puntuation before and after words
                        String processedToken = removeSpecialCharacter(stemmedToken);
                        if (processedToken.equals(""))
                            continue;
                        if (StopWords.isStopword(processedToken))
                            continue;
                        
                        if (tokenMap.containsKey(processedToken)){
                            int cnt = tokenMap.get(processedToken);
                            tokenMap.put(processedToken, cnt + 1);
                        } else{
                            tokenMap.put(processedToken, 1);
                        }
                    }
                }
            }
        }
        
        System.out.println("Complete 1st iteration....... ");
        int index  = 0;
        Map<String, Integer> tokenToIndex = new HashMap<String, Integer>();
        Map<Integer, String> indexToToken = new HashMap<Integer, String>();
        for (String token: tokenMap.keySet()){
            int cnt = tokenMap.get(token);
            if (cnt >= 30 && cnt <= 3000){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        vocab.setSentimentWords();
        
        System.out.println("Complete vocabulary creation ");
        System.out.println("******************************************");
        
        
        System.out.println("Generating documents objects using vocabulary");
        fileCount = 0;
        documents = new ArrayList<Document>();
        for (File f: files){
            List<Sentence> sentenceList = new ArrayList<Sentence>();
            Document doc = new Document();
            System.out.println("Process " + ++fileCount + "th file");
            int count = 0;
            br = new BufferedReader(new FileReader(f));
            while ((sCurrentLine = br.readLine()) != null){
                if (++count == 6){   
                    String review = sCurrentLine;
                    
                    // split each review into sentences. 
                    String[] sentences = review.split("[\\.\\!\\?\\;]");
                    for (String sentence : sentences){
                       
                        Sentence newSentence = new Sentence();
                        List<Integer> tokensInSentence = new ArrayList<Integer>();
                        String[] tokens = sentence.split("\\s");
                        for (String token : tokens){
                            PorterStemmer stem = new PorterStemmer();
                            String stemmedToken = stem.stemming(token);
                            
                            // remove puntuation before and after words
                            String processedToken = removeSpecialCharacter(stemmedToken);
                            if (processedToken.equals(""))
                                continue;
                            if (StopWords.isStopword(processedToken))
                                continue;
                            
                            if (tokenToIndex.containsKey(token)){
                                int tokenIndex = tokenToIndex.get(token);
                                tokensInSentence.add(tokenIndex);
                            }
                            
                            //System.out.println(tokensInSentence.size());
                        }
                        
                        if (tokensInSentence.size() > 7 || tokensInSentence.size() == 0)
                            continue;
                        
                        newSentence.setTokens(tokensInSentence);
                        sentenceList.add(newSentence);
                    }   
                }
            }
            
            doc.setSentences(sentenceList);
            documents.add(doc);
        }
        
        System.out.println("Successfully process all the documents");    
    }
  
   
    public DataSet(String filePath, boolean flag) throws IOException{
        BufferedReader br = null;
        String sCurrentLine = "";
        File file = new File(filePath);
        File[] files = file.listFiles();
        
        // we will count the number of tokens appears. This will be used for further filtering the words
        Map<String, Integer> tokenMap = new HashMap<String, Integer>();
        // first iteration, creating vocabulary list. 
        int fileCount = 0;
        List<String> documentStringList = new ArrayList<String>();
        for (File folder: files){
            for (File f: folder.listFiles()){
                System.out.println("Process " + ++fileCount + "th file");
                String documentString = "";
                br = new BufferedReader(new FileReader(f));
                while ((sCurrentLine = br.readLine()) != null){
                    documentString = documentString + sCurrentLine + " ";
                        String text = sCurrentLine;
                        // remove the stop words
                        String[] tokens = text.split("\\s");
                        for (String token : tokens){
                            PorterStemmer stem = new PorterStemmer();
                            String stemmedToken = stem.stemming(token);
                               
                            // remove puntuation before and after words
                            String processedToken = removeSpecialCharacter(stemmedToken);
                            if (processedToken.equals(""))
                                continue;
                            if (StopWords.isStopword(processedToken))
                                continue;
                            
                            if (tokenMap.containsKey(processedToken)){
                                int cnt = tokenMap.get(processedToken);
                                tokenMap.put(processedToken, cnt + 1);
                            } else{
                                tokenMap.put(processedToken, 1);
                            }
                        }
                    
                }
                
                documentStringList.add(documentString);
            }
        }
        
        System.out.println("Complete 1st iteration....... ");
        int index  = 0;
        Map<String, Integer> tokenToIndex = new HashMap<String, Integer>();
        Map<Integer, String> indexToToken = new HashMap<Integer, String>();
        for (String token: tokenMap.keySet()){
            int cnt = tokenMap.get(token);
            if (cnt >= 10 && cnt <= 2000){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        vocab.setSentimentWords();
        System.out.println("Complete vocabulary creation ");
        System.out.println("******************************************");
        
        
        System.out.println("Generating documents objects using vocabulary");
        fileCount = 0;
        documents = new ArrayList<Document>();
        
        for (String text  : documentStringList){
            List<Sentence> sentenceList = new ArrayList<Sentence>();
            Document doc = new Document();
            System.out.println("Process " + ++fileCount + "th file");
            String[] sentences = text.split("[\\.\\!\\?\\;]");
            
            
            for (String paper : sentences){
                Sentence newSentence = new Sentence();
                List<Integer> tokensInSentence = new ArrayList<Integer>();
                String[] tokens = paper.split("\\s");
                for (String token : tokens){
                    PorterStemmer stem = new PorterStemmer();
                    String stemmedToken = stem.stemming(token);
                    
                    // remove puntuation before and after words
                    String processedToken = removeSpecialCharacter(stemmedToken);
                    if (processedToken.equals(""))
                        continue;
                    if (StopWords.isStopword(processedToken))
                        continue;
                    
                    if (tokenToIndex.containsKey(token)){
                        int tokenIndex = tokenToIndex.get(token);
                        tokensInSentence.add(tokenIndex);
                    }
                }
                newSentence.setTokens(tokensInSentence);
                sentenceList.add(newSentence);
            }
            
            doc.setSentences(sentenceList);
            documents.add(doc);
    
        }        
        System.out.println("Successfully process all the documents");    
    }
  
    
    
    private String removeSpecialCharacter(String token){
        int start = 0; 
        int end = token.length()-1;
        for (int i = 0; i < token.length(); i++){
            char ch = token.charAt(i);
            if (!Character.isLetterOrDigit(ch))
                start++;
            else
                break;
        }
        
        for (int i = token.length()-1; i >=0; i--){
            char ch = token.charAt(i);
            if (!Character.isLetterOrDigit(ch))
                end--;
            else
                break;
        }
        
        if (start < end){
            return token.substring(start, end+1);
        } else
            return "";
        
    }
     
    public static void main(String[] args) throws IOException{
        String aa = "adfadsf.aaa,aaaa;aaaaa!aaaaaaa";
        String[] strs = aa.split("[\\.\\!\\?\\;]");
        
    }
    
}
