package ml.topicModel.sentenceLDA;

import java.util.HashSet;
import java.util.Set;

public class SentimentWords {
    private Set<String> positive;
    private Set<String> negative;
    
    public SentimentWords(){
        positive = new HashSet<String>();
        negative = new HashSet<String>();
        
        positive.add("dazzling");
        positive.add("brilliant");
        positive.add("phenomenal");
        positive.add("excellent");
        positive.add("fantastic");
        positive.add("gripping");
        positive.add("mesmerizing");
        positive.add("riveting");
        positive.add("spectacular");
        positive.add("cool");
        positive.add("awesome");
        positive.add("thrilling");
        positive.add("moving");
        positive.add("exciting");
        positive.add("love");
        positive.add("wonderful");
        positive.add("best");
        positive.add("great");
        positive.add("superb");
        positive.add("still");
        positive.add("beautiful");
        
        negative.add("sucks");
        negative.add("terrible");
        negative.add("awful");
        negative.add("unwatchable");
        negative.add("hideous");
        negative.add("bad");
        negative.add("cliched");
        negative.add("boring");
        negative.add("stupid");
        negative.add("slow");
        negative.add("worst");
        negative.add("waste");
        negative.add("unexcit");
        negative.add("rubbish");
        negative.add("tedious");
        negative.add("unbearable");
        negative.add("pointless");
        negative.add("cheesy");
        negative.add("frustrated");
        negative.add("awkward");
        negative.add("disappointing");
    }
    
    public Set<String> getPositiveWords(){
        return positive;
    }
    
    public Set<String> getNegativeWords(){
        return negative;
    }
}
