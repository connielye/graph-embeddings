package com.prime.common.computinghelper;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;

import java.util.*;

/**
 *  this clustering uses k-means to cluster relations which has many pattens. For example, citing the paper "the head-tail entities of the relation "location_location_contains" have many patterns such as country-city, country-university, continent-country and
 *  so on"
 */

public class RelationCluster {

    final static Logger logger = Logger.getLogger(RelationCluster.class);
    Helper helper;
    Map<Triple<Integer, Integer, Integer>, double[]> relationCVectors;
    ArrayList<double[]> entityVectors;
    ArrayList<Triple<Integer, Integer, Integer>> triples;
    int relationSize;
    Map<Integer, ArrayList<Pair<Integer, Integer>>> entityPairs;
    Map<double[], double[][]> matricesMap;
    Map<Integer, ArrayList<double[]>> relationCluster;


    public RelationCluster(ArrayList<double[]> entityVectors, int relationSize, ArrayList<Triple<Integer, Integer, Integer>> triples, int k, int epochs, int rows, int columns){
        this.helper = new Helper();
        this.entityVectors = entityVectors;
        this.triples = triples;
        this.relationSize = relationSize;
        this.entityPairs = generateEntityPair();
        this.relationCVectors = new HashMap<Triple<Integer, Integer, Integer>, double[]>();
        this.matricesMap = new HashMap<double[], double[][]>();
        this.relationCluster = new HashMap<Integer, ArrayList<double[]>>();
        fit(k, epochs, rows, columns);

    }

    /**
     * generate entity pairs (head-tail)
     * @return a Map of entity pairs, key: relation, value:corresponding pairList
     */

    private Map<Integer, ArrayList<Pair<Integer, Integer>>> generateEntityPair(){
        Map<Integer, ArrayList<Pair<Integer, Integer>>> entityPairs = new HashMap<Integer, ArrayList<Pair<Integer, Integer>>>();

        for (int i = 0; i < relationSize; i++){
            ArrayList<Pair<Integer, Integer>> pairList = new ArrayList<Pair<Integer, Integer>>();
            entityPairs.put(i, pairList);
        }
        for (Triple<Integer, Integer, Integer> triple : triples){
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            Pair<Integer, Integer> pair = Pair.of(head, tail);
            entityPairs.get(relation).add(pair);
        }
        return entityPairs;
    }

    /**
     * k-mean clustering for each relation using euclidean distance between center and offset(head-tail)
     * @param pairList
     * @param k
     * @param relation
     * @param epochs
     * @param rows
     * @param columns
     */
    private void clustering (ArrayList<Pair<Integer, Integer>> pairList, int k, int relation, int epochs, int rows, int columns){
        ArrayList<double[]> offsets = new ArrayList<double[]>();
        for (Pair<Integer, Integer> pair: pairList){
            double[] headVector = entityVectors.get(pair.getLeft());
            double[] tailVector = entityVectors.get(pair.getRight());
            double[] offset = new double[headVector.length];
            for (int i = 0; i< headVector.length; i++){
                offset[i] = headVector[i] - tailVector[i];
            }
            offsets.add(offset);
        }

        Random random = new Random();
        ArrayList<double[]> centerPoints = new ArrayList<double[]>();
        for (int i = 0; i < k; i++){
            double[] vector = new double[offsets.size()];
            for (int j=0; j< offsets.size(); j++){
                vector[j] = random.nextDouble();
            }
            centerPoints.add(vector);
        }
        ArrayList<ArrayList<Integer>> clusters = new ArrayList<ArrayList<Integer>>();
        for (int epoch = 0; epoch< epochs; epoch++){
            for (int m = 0; m < offsets.size(); m ++){
                double[] vector = offsets.get(m);
                double[] point0Vector = centerPoints.get(0);
                int index = 0;
                double minDistance = helper.euclideanDistance(vector, point0Vector);
                for (int n=1; n<k; n++){
                    double[] pointVector = centerPoints.get(n);
                    double eDistance = helper.euclideanDistance(vector, pointVector);
                    if(eDistance<minDistance){
                        minDistance = eDistance;
                        index = n;
                    }
                }
                clusters.get(index).add(m);
            }
            logger.info("Epoch: " + (epoch+1));
            ArrayList<double[]> newCenterPoints = new ArrayList<double[]>();
            for (int l = 0; l<clusters.size(); l++){
                ArrayList<Integer> group = clusters.get(l);
                ArrayList<double[]> points = new ArrayList<double[]>();
                for(int index: group){
                    points.add(offsets.get(index));
                }
                double[] centerPoint = helper.centerPoint(points);
                newCenterPoints.add(centerPoint);
            }
            if(centerPoints != newCenterPoints){
                centerPoints = newCenterPoints;
            } else {
                break; // break if converged.
            }
        }
        int relationC = 0;
        for(ArrayList<Integer> cluster: clusters){
            relationC ++;
            for(int index: cluster){
                Triple<Integer, Integer, Integer> triple = triples.get(index);
                double[] normCenterPoint = helper.norm(centerPoints.get(index));
                relationCVectors.put(triple, normCenterPoint);
            }
        }
        for (double[] centerPoint: centerPoints){
            double[][] matrix = helper.identityMatrix(rows, columns);
            matricesMap.put(centerPoint, matrix);
        }
        relationCluster.put(relation, centerPoints);

    }

    /**
     * learning for the whole dataset
     * @param k
     * @param epochs
     * @param rows
     * @param columns
     */

    private void fit (int k, int epochs, int rows, int columns){
        for (int i = 0; i < relationSize; i++){
            ArrayList<Pair<Integer, Integer>> pairList = entityPairs.get(i);
            clustering(pairList, k, i, epochs, rows, columns);
        }
    }

    public Map<Triple<Integer, Integer, Integer>, double[]> getRelationCVectors() {
        return relationCVectors;
    }

    public Map<double[], double[][]> getMatricesMap() {
        return matricesMap;
    }

    public Map<Integer, ArrayList<double[]>> getRelationCluster() {
        return relationCluster;
    }
}
