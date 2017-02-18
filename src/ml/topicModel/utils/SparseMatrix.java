package ml.topicModel.utils;


import java.util.HashMap;
import java.util.Map;

public class SparseMatrix {
    
    public class SparseVector{
        public Map<Integer, Integer> vec;
        
        public SparseVector(){
            vec = new HashMap<Integer, Integer>();
        }
    }
    private int N;
    
    SparseVector[] sparseVectors;
    
    public SparseMatrix(int N){
        sparseVectors = new SparseVector[N];
        for (int i = 0; i < N; i++){
            sparseVectors[i] = new SparseVector();
        }
    }
    
    
    public int get(int i, int j){
        SparseVector vector = sparseVectors[i];
        if (vector.vec.containsKey(j)){
            return vector.vec.get(j);
        } else{
            return 0;
        }
    }
    
    public void increment(int i, int j){
        SparseVector vector = sparseVectors[i];
        if (vector.vec.containsKey(j)){
            int currValue = vector.vec.get(j);
            vector.vec.put(j, (currValue+1));
        } else{
            vector.vec.put(j,  1);
        }
    }
    
    public void decrement(int i, int j){
        SparseVector vector = sparseVectors[i];
        int currValue = vector.vec.get(j);
        if (currValue == 1){
            vector.vec.remove(j);
        } else{
            vector.vec.put(j, (currValue-1));
        }
    }
    
   
    
  
}
