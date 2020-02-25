package com.prime.common.similarity;

import java.util.ArrayList;

/**
 * calcualte cosine com.prime.common.similarity between two vectors
 */

public class CosineSimilarity {

    double similarityScore;
    ArrayList<Double> vector1;
    ArrayList<Double> vector2;

    public CosineSimilarity(ArrayList<Double> vector1, ArrayList<Double> vector2){
        this.vector1 = vector1;
        this.vector2 = vector2;
        this.similarityScore = calculateCosine();
    }

    private double calculateCosine(){
        double sumNumerator = 0d;
        double sumSquare1 = 0d;
        double sumSquare2 = 2d;
        for (int i = 0; i<vector1.size(); i ++){
            double numerator = vector1.get(i) * vector2.get(i);
            double square1 = Math.pow(vector1.get(i), 2);
            double square2 = Math.pow(vector2.get(i), 2);
            sumNumerator = sumNumerator + numerator;
            sumSquare1 = sumSquare1 + square1;
            sumSquare2 = sumSquare2 + square2;
        }
        double score = sumNumerator / (Math.sqrt(sumSquare1)* Math.sqrt(sumSquare2));
        return score;
    }

    public double getSimilarityScore() {
        return similarityScore;
    }
}
