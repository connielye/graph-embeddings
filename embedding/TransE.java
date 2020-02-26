package com.prime.common.embedding;

import com.prime.common.computinghelper.Helper;
import com.prime.common.computinghelper.NegativeSampling;
import com.prime.common.io.*;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.*;

/**
 * TransE algorithm -- basic graph embedding
 *
 */

public class TransE {

    final static Logger logger = Logger.getLogger(TransE.class);
    Helper helper;
    NegativeSampling sample;
    ArrayList<double[]> relationVectors;
    ArrayList<double[]> entityVectors;
    public TransE(){
        this.helper = new Helper();
        this.sample = new NegativeSampling();
    }

    /**
     * initialize relationVectors and entityVectors
     * @param k
     */

    private void initialize (int entitySize, int relationSize, int k){
        relationVectors = new ArrayList<double[]>();
        entityVectors = new ArrayList<double[]>();
        for (int i = 0; i < relationSize; i ++){
            double[] lVector = helper.initVector(k);
            double[] normedlVector = helper.norm(lVector);
            relationVectors.add(normedlVector);
        }

        for (int j = 0; j < entitySize; j++){
            double[] eVector = helper.initVector(k);
            double[] normedeVector = helper.norm(eVector);
            entityVectors.add(normedeVector);
        }
    }


    /**
     * sgd
     * @param tBatch
     * @param margin
     * @param learningRate
     * @param L1
     * @param k
     */

    private void update(ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> tBatch, double margin, double learningRate, Boolean L1, int k){
       ArrayList<double[]> copyEntityVectors = (ArrayList<double[]>) entityVectors.clone();
        ArrayList<double[]> copyRelationVectors = (ArrayList<double[]>) relationVectors.clone();

        for (int i = 0; i<tBatch.size(); i++){
            int head = tBatch.get(i).getLeft().getLeft();
            int label = tBatch.get(i).getLeft().getMiddle();
            int tail = tBatch.get(i).getLeft().getRight();
            double[] headVector = copyEntityVectors.get(head);
            double[] labelVector = copyRelationVectors.get(label);
            double[] tailVector = copyEntityVectors.get(tail);
            double distanceL1 = helper.distanceL1(headVector, labelVector, tailVector, k);
            double distanceL2 = helper.distanceL2(headVector, labelVector, tailVector, k);

            int headC = tBatch.get(i).getRight().getLeft();
            int labelC = tBatch.get(i).getRight().getMiddle();
            int tailC = tBatch.get(i).getRight().getRight();
            double[] headCVector = copyEntityVectors.get(headC);
            double[] labelCVector = copyRelationVectors.get(labelC);
            double[] tailCVector = copyEntityVectors.get(tailC);
            double distanceCL1 = helper.distanceL1(headCVector, labelCVector, tailCVector, k);
            double distanceCL2 = helper.distanceL2(headCVector, labelCVector, tailCVector, k);

            ArrayList<Double> deltaVector = new ArrayList<Double>();
            ArrayList<Double> deltaCVector = new ArrayList<Double>();
            if (L1 == true ){
                double lossL1 = margin + distanceL1 - distanceCL1;
                if (lossL1 > 0){
                    for (int j = 0; j < k; j++){
                        double temp = tailVector[j] - headVector[j] - labelVector[j];
                        double delta;
                        double deltaC;
                        if (temp >= 0){
                            delta = 1d; // first step chain rule
                            deltaVector.add(delta);
                        } else{
                            delta = -1d; // first step chain rule
                            deltaVector.add(delta);
                        }
                        double tempC = tailCVector[j] - headCVector[j] - labelCVector[j];
                        if (tempC >= 0){
                            deltaC = 1d; // first step chain rule
                            deltaCVector.add(deltaC);
                        } else {
                            deltaC = -1d; // first step chain rule
                            deltaCVector.add(deltaC);
                        }
                    }
                }
            } else {
                double lossL2 = margin + distanceL2 - distanceCL2;
                if (lossL2 > 0) {
                    for (int m = 0; m < k; m++) {
                        double delta = 2 * (tailVector[m] - headVector[m] - labelVector[m]); // first step chain rule
                        double deltaC = 2 * (tailCVector[m] - headCVector[m] - labelCVector[m]); // first step chain rule
                        deltaVector.add(delta);
                        deltaCVector.add(deltaC);
                    }
                }
            }
            if (deltaVector.size() != 0) {
                for (int n = 0; n < k; n++) {
                    double newHead = headVector[n] + learningRate * deltaVector.get(n);
                    double newLable = labelVector[n] + learningRate * (deltaVector.get(n) - deltaCVector.get(n));
                    double newTail = tailVector[n] - learningRate * deltaVector.get(n);
                    headVector[n] = newHead;
                    labelVector[n] = newLable;
                    tailVector[n] = newTail;
                }
                double[] normedHeadVector = helper.norm(headVector);
                double[] normedLabelVector = helper.norm(labelVector);
                double[] normedTailVector = helper.norm(tailVector);
                copyEntityVectors.set(head, normedHeadVector);
                copyRelationVectors.set(label, normedLabelVector);
                copyEntityVectors.set(tail, normedTailVector);
            }
        }
        entityVectors = copyEntityVectors;
        relationVectors = copyRelationVectors;
    }


    /**
     * validate the result on dev set after each epoch
     * @param devTriple
     * @return accuracy
     */

    private double validation(ArrayList<Triple<Integer, Integer, Integer>> devTriple, Boolean L1, int k, double margin){
        int count = 0;
        for (int i = 0; i < devTriple.size(); i++){
            int headId = devTriple.get(i).getLeft();
            int labelId = devTriple.get(i).getMiddle();
            int tailId = devTriple.get(i).getRight();
            double[] headVector = entityVectors.get(headId);
            double[] labelVector = relationVectors.get(labelId);
            double[] tailVector = entityVectors.get(tailId);
            double distance;
            if (L1 == true){
                distance = helper.distanceL1(headVector, labelVector, tailVector, k);
            } else {
                distance = helper.distanceL2(headVector, labelVector, tailVector, k);
            }
            if (distance < margin){
                count ++;
            }
        }
        double accuracy = count / devTriple.size() * 100;
        return accuracy;
    }

    /**
     * save serialized trained model in local disk after learning
     * @param entityOutput
     * @throws IOException
     */

    private void saveModel(String entityOutput, String relationOutput, Map<String, Integer> entity2Id, Map<String, Integer> relation2Id){
        Map<String, double[]> entityMap = new HashMap<String, double[]>();
        Map<String, double[]> relationMap = new HashMap<String, double[]>();
        Iterator iteratorEntity = entity2Id.entrySet().iterator();
        while (iteratorEntity.hasNext()){
            Map.Entry entry = (Map.Entry)iteratorEntity.next();
            String entityName = entry.getKey().toString();
            int entityId = Integer.parseInt(entry.getValue().toString());
            entityMap.put(entityName, entityVectors.get(entityId));
        }
        Iterator iteratorRelation = relation2Id.entrySet().iterator();
        while (iteratorRelation.hasNext()){
            Map.Entry entry = (Map.Entry)iteratorRelation.next();
            String relationName = entry.getKey().toString();
            int relationId = Integer.parseInt(entry.getValue().toString());
            relationMap.put(relationName, relationVectors.get(relationId));
        }

        SerializeModelVectors modelEntity = new SerializeModelVectors(entityMap);
        WriteModel writer = new WriteModel();
        writer.write(entityOutput, modelEntity);

        SerializeModelVectors modelRelation = new SerializeModelVectors(relationMap);
        writer.write(relationOutput, modelRelation);


    }

    /**
     * learning phase and cross validation
     * @param trainTriple
     * @param devTriple
     * @param learningRate
     * @param margin
     * @param k
     * @param L1
     * @param batchSize
     * @param epochs
     * @param entityOutput
     * @param relationOutput
     * @throws IOException
     */
    public void learn(ArrayList<Triple<Integer, Integer, Integer>> trainTriple, ArrayList<Triple<Integer, Integer, Integer>> devTriple, int entitySize, int relationSize, Map<String, Integer> entity2Id,
                      Map<String, Integer> relation2Id, double learningRate, double margin, int k, Boolean L1, int batchSize, int epochs, String entityOutput, String relationOutput) throws IOException{
        initialize(entitySize, relationSize, k);
        for (int epoch =0; epoch < epochs; epoch ++){
            int trainSize = trainTriple.size();
            int batchNumber = Math.round(trainSize/batchSize);

            for (int i = 0; i < batchNumber; i ++){
                ArrayList<Triple<Integer, Integer, Integer>> sBatch = helper.sample(trainTriple, batchSize);
                ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> tBatch = sample.generateUnif(sBatch, entitySize);
                update(tBatch, margin, learningRate, L1, k);
            }
            logger.info("Epoch: " + epoch);

            if(devTriple != null){
                double accuracy = validation(devTriple, L1, k, margin);
                logger.info("Validation Accuracy: " + accuracy + "; Epoch: " + epoch);
            }
        }
        saveModel(entityOutput, relationOutput, entity2Id, relation2Id);

    }


}
