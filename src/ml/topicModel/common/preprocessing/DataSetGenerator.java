package ml.topicModel.common.preprocessing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.topicModel.common.data.DataSet;
import ml.topicModel.common.data.Document;
import ml.topicModel.common.data.SDocument;
import ml.topicModel.common.data.Sentence;
import ml.topicModel.common.data.Vocabulary;
import ml.topicModel.common.data.WDocument;

public class DataSetGenerator {
    
    public static int minCnt = 10;
    public static int maxCnt = 5000;
    // for review data sets
    public static DataSet createYelpDataSetForSentenceLevel(String filePath) throws IOException{
        
        DataSet dataset  = new DataSet();
        dataset.dataSetName = "yelp";
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
                        if (containsInvalidCharacter(processedToken)){
                            continue;
                        }
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
            if (cnt >= minCnt && cnt <= maxCnt){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        Vocabulary vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        vocab.setSentimentWords();
        
        System.out.println("Complete vocabulary creation ");
        System.out.println("******************************************");
        
        System.out.println("Generating documents objects using vocabulary");
        fileCount = 0;
        List<Document> documents = new ArrayList<Document>();
        for (File f: files){
            List<Sentence> sentenceList = new ArrayList<Sentence>();
            Document doc = new SDocument();
            System.out.println("Process " + ++fileCount + "th file");
            int count = 0;
            br = new BufferedReader(new FileReader(f));
            while ((sCurrentLine = br.readLine()) != null){
                count++;
                if (count == 3){
                    String[] ratings = sCurrentLine.split(":");
                    double rating = Double.parseDouble(ratings[1].trim());
                    doc.setRating(rating);
                }
                
                if (count == 6){   
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
                            if (containsInvalidCharacter(processedToken)){
                                continue;
                            }
                            if (StopWords.isStopword(processedToken))
                                continue;
                            
                            if (tokenToIndex.containsKey(token)){
                                int tokenIndex = tokenToIndex.get(token);
                                tokensInSentence.add(tokenIndex);
                            }
                        }
                        
                        newSentence.setTokens(tokensInSentence);
                        if (newSentence.getTokens().size() >= 1 && newSentence.getTokens().size() < 9){
                            sentenceList.add(newSentence);
                        }
                        
                    }   
                }
            }
            
            ((SDocument)doc).setSentences(sentenceList);
            documents.add(doc);
        }
        
        dataset.setDocuments(documents);
        dataset.setNumDocuments(documents.size());
        dataset.setVocabulary(vocab);
        
        System.out.println("Successfully process all the documents");    
        
        return dataset;
    }
       
    public static DataSet createNIPSDataSetForSentenceLevel(String filePath) throws IOException{    
        DataSet dataset = new DataSet();
        dataset.dataSetName = "nips";
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
            if (cnt >= minCnt && cnt <= maxCnt){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        Vocabulary vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        vocab.setSentimentWords();
        System.out.println("Complete vocabulary creation ");
        System.out.println("******************************************");
        
        System.out.println("Generating documents objects using vocabulary");
        fileCount = 0;
        List<Document> documents = new ArrayList<Document>();
        
        for (String text  : documentStringList){
            List<Sentence> sentenceList = new ArrayList<Sentence>();
            Document doc = new SDocument();
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
            
            ((SDocument)doc).setSentences(sentenceList);
            documents.add(doc);
        }   
        
        dataset.setDocuments(documents);
        dataset.setNumDocuments(documents.size());
        dataset.setVocabulary(vocab);
        
        System.out.println("Successfully process all the documents");
        
        return dataset;
    }
       
    public static DataSet createYelpDataSetForWordLevel(String filePath) throws IOException{
        DataSet dataset = new DataSet();
        dataset.dataSetName = "yelp";
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
            if (cnt >= minCnt && cnt <= maxCnt){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        Vocabulary vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        vocab.setSentimentWords();
        System.out.println("Complete vocabulary creation ");
        System.out.println("******************************************");
        
        
        System.out.println("Generating documents objects using vocabulary");
        fileCount = 0;
        List<Document> documents = new ArrayList<Document>();
        for (File f: files){
            List<Integer> tokensInDoc = new ArrayList<Integer>();
            Document doc = new WDocument();
            System.out.println("Process " + ++fileCount + "th file");
            int count = 0;
            br = new BufferedReader(new FileReader(f));
            while ((sCurrentLine = br.readLine()) != null){
                count++;
                if (count == 3){
                    String[] ratings = sCurrentLine.split(":");
                    double rating = Double.parseDouble(ratings[1].trim());
                    doc.setRating(rating);
                }
                
                if (count == 6){   
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
                        
                        if (tokenToIndex.containsKey(token)){
                            int tokenIndex = tokenToIndex.get(token);
                            tokensInDoc.add(tokenIndex);
                        }
                    }
                }
            }
            
            ((WDocument)doc).setTokens(tokensInDoc);
            documents.add(doc);
        }
        
        dataset.setDocuments(documents);
        dataset.setNumDocuments(documents.size());
        dataset.setVocabulary(vocab);
        
        System.out.println("Successfully process all the documents");  
        
        return dataset;
    }
  
    
    public static DataSet createNIPSDataSetForWordLevel(String filePath) throws IOException{
        DataSet dataset = new DataSet();
        dataset.dataSetName = "nips";
        BufferedReader br = null;
        String sCurrentLine = "";
        File file = new File(filePath);
        File[] files = file.listFiles();
        
        // we will count the number of tokens appears. This will be used for further filtering the words
        Map<String, Integer> tokenMap = new HashMap<String, Integer>();
        // first iteration, creating vocabulary list. 
        int fileCount = 0;
        
        for (File folder: files){
            for (File f: folder.listFiles()){
                System.out.println("Process " + ++fileCount + "th file");
                
                br = new BufferedReader(new FileReader(f));
                while ((sCurrentLine = br.readLine()) != null){
                   
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
            }
        }
        
        System.out.println("Complete 1st iteration....... ");
        int index  = 0;
        Map<String, Integer> tokenToIndex = new HashMap<String, Integer>();
        Map<Integer, String> indexToToken = new HashMap<Integer, String>();
        for (String token: tokenMap.keySet()){
            int cnt = tokenMap.get(token);
            if (cnt >= minCnt && cnt <= maxCnt){
                tokenToIndex.put(token, index);
                indexToToken.put(index, token);
                index++;
            }
        }
        
        Vocabulary vocab = new Vocabulary();
        vocab.setIndexTotokenMap(indexToToken);
        vocab.settokenToIndex(tokenToIndex);
        System.out.println("Complete vocabulary creation ");
        System.out.println("******************************************");
        
        
        System.out.println("Generating documents objects using vocabulary");
        fileCount = 0;
        List<Document> documents = new ArrayList<Document>();
        
        for (File folder: files){
            for (File f: folder.listFiles()){
                List<Integer> tokensInDoc = new ArrayList<Integer>();
                Document doc = new WDocument();
                System.out.println("Process " + ++fileCount + "th file");
               
                br = new BufferedReader(new FileReader(f));
                while ((sCurrentLine = br.readLine()) != null){
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
                            
                            if (tokenToIndex.containsKey(token)){
                                int tokenIndex = tokenToIndex.get(token);
                                tokensInDoc.add(tokenIndex);
                            }
                        }
                    
                }
                
                ((WDocument)doc).setTokens(tokensInDoc);
                documents.add(doc);
            } 
        }
        
        dataset.setDocuments(documents);
        dataset.setNumDocuments(documents.size());
        dataset.setVocabulary(vocab);
        System.out.println("Successfully process all the documents");   
        
        return dataset;
    }
    
    
    public static String removeSpecialCharacter(String token){
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
    
    public static boolean containsInvalidCharacter(String token){
      
        if (token.length() < 3){
            return true;
        }
        
        int count = 0;
        for (int i = 0; i < token.length(); i++){
            char ch = token.charAt(i);
            if (Character.isDigit(ch)){
                return true;
            }
            
            if (Character.isLetter(ch)){
                count++;
            }
        }
        
        if (count < 3){
            return true;
        }
        
        return false;
    }
    
}
