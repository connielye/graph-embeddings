package com.prime.common.embedding;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.distribution.UniformRealDistribution;

/**
 * help functions for computing
 *
 */
public class ComputeHelper {

    /**
     * initialize vectors from (-6/sqr k, 6/sqr k) within uniform distribution
     * @param k for com.prime.common.embedding dimensions
     * @return initialized vector
     */
    public ArrayList<Double> initVector(int k){
        ArrayList<Double> vector = new ArrayList<Double>();
        for (int i = 0; i < k; i++) {
            double low = -(6 / Math.sqrt(k));
            double high = 6 / Math.sqrt(k);
            double random = new UniformRealDistribution(low, high).sample();
            vector.add(random);
        }
        return vector;

    }

    /**
     * normlize vectors the norm of entity and relation is 1
     * @param vector
     * @return normlized vectors
     */
    public ArrayList<Double> norm(ArrayList<Double> vector){
        ArrayList<Double> result = new ArrayList<Double>();
        double sum = 0;
        for (double number: vector){
            sum = sum + Math.pow(number, 2d);
        }

        double demoniator = Math.sqrt(sum);

        for (int i = 0; i < vector.size(); i++){
            double normNumber = vector.get(i) / demoniator;
            result.add(normNumber);
        }
        return result;
    }

    /**
     * compute distance with L1 regulation
     * @param head
     * @param label
     * @param tail
     * @param k
     * @return distance with L1 regulation
     */

    public Double distanceL1(ArrayList<Double> head, ArrayList<Double> label, ArrayList<Double> tail, int k){
        double sum = 0;
        for (int i = 0; i<k; i++){
            double calNumber = tail.get(i) - head.get(i) - label.get(i);
            sum = sum + Math.abs(calNumber);
        }
        return sum;
    }

    /**
     * compute distance with L2 regulation
     * @param head
     * @param label
     * @param tail
     * @param k
     * @return distance with L2 regulation
     */

    public Double distanceL2(ArrayList<Double> head, ArrayList<Double> label, ArrayList<Double> tail, int k){
        double sum = 0;
        for (int i = 0; i < k; i++){
            double calNumber = tail.get(i) - head.get(i) - label.get(i);
            sum = sum + calNumber * calNumber;
        }
        return sum;
    }

    /**
     * sample a minibatch of size size
     * @param triples
     * @param size
     * @return minibatch for sgd
     */

    public ArrayList<Triple<Integer, Integer, Integer>> sample( ArrayList<Triple<Integer, Integer, Integer>> triples, int size){
        ArrayList<Triple<Integer, Integer, Integer>> sBatch = new ArrayList<Triple<Integer, Integer, Integer>>();
        ArrayList<Integer> randomIndex = new ArrayList<Integer>();
        Random random = new Random();
        int tripleSize = triples.size();
        for (int i = 0; i < size; i++){
            int index = random.nextInt(tripleSize);
            while (! randomIndex.contains(index)){
                randomIndex.add(index);
            }
        }
        for (int idx: randomIndex){
            sBatch.add(triples.get(idx));
        }
        return sBatch;
    }


}
