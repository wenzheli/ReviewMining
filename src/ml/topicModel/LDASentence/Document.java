package ml.topicModel.jointAspectSentiment;

import java.util.List;

public class Document{
    List<Integer> tokens;
    double rating;
    
    public void setRating(double rating){
        this.rating = rating;
    }
    
    public void setTokens(List<Integer> tokens){
        this.tokens = tokens;
    }
    
    public int getNumOfTokens(){
        return tokens.size();
    }
    public List<Integer> getTokens(){
        return tokens;
    }
    
    public int getToken(int index){
        return tokens.get(index);
    }
    
    public double getRating(){
        return rating;
    }
}
