package com.prime.common.similarity;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * calculate jaccard coeffiency among two string
 */

public class JaccardSimilarity {

    Double similaryScore;
    ArrayList<String> inputList1;
    ArrayList<String> inputList2;

    public JaccardSimilarity(ArrayList<String> inputList1, ArrayList<String> inputList2){
        this.inputList1 = inputList1;
        this.inputList1 = inputList2;
        this.similaryScore = calculateJaccard();
    }

    private Double calculateJaccard(){
        double score;
        Set<String> intersection = new HashSet<>();
        Set<String> union = new HashSet<>();
        union.addAll(inputList1);
        union.addAll(inputList2);
        for (int i = 0; i < inputList1.size(); i++){
            String tok1 = inputList1.get(i);
            for (int j = 0; j< inputList2.size(); j++){
                String tok2 = inputList2.get(j);
                if (tok1.equals(tok2)){
                    intersection.add(tok2);
                }
            }
        }
        score = intersection.size()/ union.size();
        return score;
    }

    public Double getSimilaryScore() {
        return similaryScore;
    }
}
