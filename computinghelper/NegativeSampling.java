package com.prime.common.computinghelper;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import java.util.*;

public class NegativeSampling {

    /**
     * generate Tbatch which contains pair triple and its corrupted triple
     * @param triples
     * @return Tbatch
     */

    public ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> generateUnif(ArrayList<Triple<Integer, Integer, Integer>> triples, int entitySize){
        ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> tBatch = new ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>>();
        Random random = new Random();
        for (int i = 0; i < triples.size(); i++){
            Triple<Integer, Integer, Integer> triple = triples.get(i);
            Boolean flag = random.nextBoolean();
            if(flag == true){
                int head = triple.getLeft();
                int randEntity = random.nextInt(entitySize);
                while (randEntity == head){
                    randEntity = random.nextInt(entitySize);
                }
                Triple<Integer, Integer, Integer> corruptedTriple = Triple.of(randEntity, triple.getMiddle(), triple.getRight());
                Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>> pair = Pair.of(triple, corruptedTriple);
                tBatch.add(pair);
            }
            if(flag == false){
                int tail = triple.getRight();
                int randEntity = random.nextInt(entitySize);
                while (randEntity == tail){
                    randEntity = random.nextInt(entitySize);
                }
                Triple<Integer, Integer, Integer> corruptedTriple = Triple.of(triple.getLeft(), triple.getMiddle(), randEntity);
                Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>> pair = Pair.of(triple, corruptedTriple);
                tBatch.add(pair);
            }
        }
        return tBatch;
    }
    /**
     * get proportion of relations with statistic heads per tail or tails per head
     * @param tripleList
     * @return Map with key relationId and value Pair of tph and hpt proportions
     */

    private Map<Integer, Pair<Double, Double>> headTailProportion(ArrayList<Triple<Integer, Integer, Integer>> tripleList, int relationSize){
        Map<Integer, Pair<Double, Double>> relationProbMap = new HashMap<Integer, Pair<Double, Double>>();
        for (int i = 0; i < relationSize; i ++){
            Set<Integer> headSet = new HashSet<Integer>();
            Set<Integer> tailSet = new HashSet<Integer>();
            for (int j = 0; j < tripleList.size(); j ++){
                int relation = tripleList.get(j).getMiddle();
                if (relation == j){
                    int head = tripleList.get(j).getLeft();
                    int tail = tripleList.get(j).getRight();
                    headSet.add(head);
                    tailSet.add(tail);
                }
            }
            double tph = tailSet.size()/headSet.size();
            double hpt = headSet.size()/tailSet.size();
            double tphProb = tph / tph + hpt;
            double hptProb = hpt / tph + hpt;
            relationProbMap.put(i, Pair.of(tphProb, hptProb));
        }
        return relationProbMap;
    }

    /**
     * generate corrupted triples based on bernoulli distribution with parameters head-tail propoertion
     * @param miniBatch
     * @return a list of corruptedTriples
     */
    public ArrayList<Triple<Integer, Integer, Integer>> generateBern(ArrayList<Triple<Integer, Integer, Integer>> miniBatch, int entitySize,
                                                                                 ArrayList<Triple<Integer, Integer, Integer>> tripleList, int relationSize){
        Map<Integer, Pair<Double, Double>> relationProbMap = headTailProportion(tripleList, relationSize);
        ArrayList<Triple<Integer, Integer, Integer>> corruptedTriples = new ArrayList<Triple<Integer, Integer, Integer>>();
        Random random = new Random();
        for (Triple<Integer, Integer, Integer> triple: miniBatch){
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double tphProb = relationProbMap.get(relation).getLeft();
            double hptProb = relationProbMap.get(relation).getRight();
            if(tphProb >= hptProb){ // the paper uses Bernoulli distribution with parameters of tphProb. I just compare the two probabilities
                int corruptedHead = random.nextInt(entitySize);
                while(corruptedHead == head){
                    corruptedHead = random.nextInt(entitySize);
                }
                Triple<Integer, Integer, Integer> corruptedTriple = Triple.of(corruptedHead, relation, tail);
                corruptedTriples.add(corruptedTriple);
            }
            else {
                int corruptedTail = random.nextInt(entitySize);
                while(corruptedTail == tail){
                    corruptedTail = random.nextInt(entitySize);
                }
                Triple<Integer, Integer, Integer> corruptedTriple = Triple.of(head, relation, corruptedTail);
                corruptedTriples.add(corruptedTriple);
            }
        }
        return corruptedTriples;
    }
}
