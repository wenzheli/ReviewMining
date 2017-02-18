package ml.topicModel.common.data;

import java.util.List;

public class WDocument extends Document{
    List<Integer> tokens;
    
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
}
