package com.prime.common.embedding;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.json.simple.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Logger;

/**
 * TransE algorithm
 *
 */

public class TransE {

    final static Logger logger = Logger.getLogger(TransE.class.getName());
    ComputeHelper helper;
    ArrayList<Integer> entityList;
    ArrayList<Integer> relationList;
    ArrayList<ArrayList<Double>> relationVectors;
    ArrayList<ArrayList<Double>> entityVectors;

    public TransE(ArrayList<Integer> entityList, ArrayList<Integer> relationList){
        this.helper = new ComputeHelper();
        this.entityList = entityList;
        this.relationList = relationList;

    }

    /**
     * initialize relationVectors and entityVectors
     * @param k
     */

    private void initialize (int k){
        relationVectors = new ArrayList<ArrayList<Double>>();
        entityVectors = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < relationList.size(); i ++){
            ArrayList<Double> lVector = helper.initVector(k);
            ArrayList<Double> normedlVector = helper.norm(lVector);
            relationVectors.add(normedlVector);
        }

        for (int j = 0; j < entityList.size(); j++){
            ArrayList<Double> eVector = helper.initVector(k);
            ArrayList<Double> normedeVector = helper.norm(eVector);
            entityVectors.add(normedeVector);
        }
    }

    /**
     * generate Tbatch which contains pair triple and its corrupted triple
     * @param triples
     * @return Tbatch
     */

    private ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> generateTBatch(ArrayList<Triple<Integer, Integer, Integer>> triples){
        ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> tBatch = new ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>>();
        Random random = new Random();
        for (int i = 0; i < triples.size(); i++){
            Triple<Integer, Integer, Integer> triple = triples.get(i);
            Boolean flag = random.nextBoolean();
            if(flag == true){
                int head = triple.getLeft();
                int randEntity = random.nextInt(entityList.size());
                while (randEntity == head){
                    randEntity = random.nextInt(entityList.size());
                }
                Triple<Integer, Integer, Integer> corruptedTriple = Triple.of(randEntity, triple.getMiddle(), triple.getRight());
                Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>> pair = Pair.of(triple, corruptedTriple);
                tBatch.add(pair);
            }
            if(flag == false){
                int tail = triple.getRight();
                int randEntity = random.nextInt(entityList.size());
                while (randEntity == tail){
                    randEntity = random.nextInt(entityList.size());
                }
                Triple<Integer, Integer, Integer> corruptedTriple = Triple.of(triple.getLeft(), triple.getMiddle(), randEntity);
                Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>> pair = Pair.of(triple, corruptedTriple);
                tBatch.add(pair);
            }
        }
        return tBatch;
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
        ArrayList<ArrayList<Double>> copyEntityVectors = (ArrayList<ArrayList<Double>>) entityVectors.clone();
        ArrayList<ArrayList<Double>> copyRelationVectors = (ArrayList<ArrayList<Double>>) relationVectors.clone();

        for (int i = 0; i<tBatch.size(); i++){
            int head = tBatch.get(i).getLeft().getLeft();
            int label = tBatch.get(i).getLeft().getMiddle();
            int tail = tBatch.get(i).getLeft().getRight();
            ArrayList<Double> headVector = copyEntityVectors.get(head);
            ArrayList<Double> labelVector = copyRelationVectors.get(label);
            ArrayList<Double> tailVector = copyEntityVectors.get(tail);
            double distanceL1 = helper.distanceL1(headVector, labelVector, tailVector, k);
            double distanceL2 = helper.distanceL2(headVector, labelVector, tailVector, k);

            int headC = tBatch.get(i).getRight().getLeft();
            int labelC = tBatch.get(i).getRight().getMiddle();
            int tailC = tBatch.get(i).getRight().getRight();
            ArrayList<Double> headCVector = copyEntityVectors.get(headC);
            ArrayList<Double> labelCVector = copyRelationVectors.get(labelC);
            ArrayList<Double> tailCVector = copyEntityVectors.get(tailC);
            double distanceCL1 = helper.distanceL1(headCVector, labelCVector, tailCVector, k);
            double distanceCL2 = helper.distanceL2(headCVector, labelCVector, tailCVector, k);

            ArrayList<Double> calculusVector = new ArrayList<Double>();
            ArrayList<Double> calculusCVector = new ArrayList<Double>();
            if (L1 == true ){
                double lossL1 = margin + distanceL1 - distanceCL1;
                if (lossL1 > 0){
                    for (int j = 0; j < k; j++){
                        double temp = tailVector.get(j) - headVector.get(j) - labelVector.get(j);
                        double calculus ;
                        double calculusC;
                        if (temp >= 0){
                            calculus = 1d;
                            calculusVector.add(calculus);
                        } else{
                            calculus = -1d;
                            calculusVector.add(calculus);
                        }
                        double tempC = tailCVector.get(j) - headCVector.get(j) - labelCVector.get(j);
                        if (tempC >= 0){
                            calculusC = 1d;
                            calculusCVector.add(calculusC);
                        } else {
                            calculusC = 0d;
                            calculusCVector.add(calculusC);
                        }
                    }
                }
            } else {
                double lossL2 = margin + distanceL2 - distanceCL2;
                if (lossL2 > 0) {
                    for (int m = 0; m < k; m++) {
                        double calculus = 2 * (tailVector.get(m) - headVector.get(m) - labelVector.get(m));
                        double calculusC = 2 * (tailCVector.get(m) - headCVector.get(m) - labelCVector.get(m));
                        calculusVector.add(calculus);
                        calculusCVector.add(calculusC);
                    }
                }
            }

            for (int n = 0; n < k; n ++){
                double newHead = headVector.get(n) + learningRate * calculusVector.get(n);
                double newLable = labelVector.get(n) + learningRate * (calculusVector.get(n) - calculusCVector.get(n));
                double newTail = tailVector.get(n) - learningRate * calculusVector.get(n);
                headVector.set(n, newHead);
                labelVector.set(n, newLable);
                tailVector.set(n, newTail);
            }
            ArrayList<Double> normedHeadVector = helper.norm(headVector);
            ArrayList<Double> normedLabelVector = helper.norm(labelVector);
            ArrayList<Double> normedTailVector = helper.norm(tailVector);
            copyEntityVectors.set(head, normedHeadVector);
            copyRelationVectors.set(label, normedLabelVector);
            copyEntityVectors.set(tail, normedTailVector);
        }
        entityVectors = copyEntityVectors;
        relationVectors = copyRelationVectors;
    }

    /**
     * learning vectors
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
    public void learn(ArrayList<Triple<Integer, Integer, Integer>> trainTriple, ArrayList<Triple<Integer, Integer, Integer>> devTriple, double learningRate, double margin,
                      int k, Boolean L1, int batchSize, int epochs, String entityOutput, String relationOutput) throws IOException{
        initialize(k);
        for (int epoch =0; epoch < epochs; epoch ++){
            int trainSize = trainTriple.size();
            int batchNumber = Math.round(trainSize/batchSize);

            for (int i = 0; i < batchNumber; i ++){
                ArrayList<Triple<Integer, Integer, Integer>> sBatch = helper.sample(trainTriple, batchSize);
                ArrayList<Pair<Triple<Integer, Integer, Integer>, Triple<Integer, Integer, Integer>>> tBatch = generateTBatch(sBatch);
                update(tBatch, margin, learningRate, L1, k);
            }
            logger.info("Epoch: " + epoch);

            if(devTriple != null){
                double accuracy = validation(devTriple, L1, k, margin);
                logger.info("Validation Accuracy: " + accuracy + "; Epoch: " + epoch);
            }
        }
        writeEntityVector(entityOutput);
        writeRelationVector(relationOutput);

    }

    /**
     * validate the result on dev set after each epoch
     * @param devTriple
     * @return accuracy
     */

    private Double validation(ArrayList<Triple<Integer, Integer, Integer>> devTriple, Boolean L1, int k, double margin){
        int count = 0;
        for (int i = 0; i < devTriple.size(); i++){
            int headId = devTriple.get(i).getLeft();
            int labelId = devTriple.get(i).getMiddle();
            int tailId = devTriple.get(i).getRight();
            ArrayList<Double> headVector = entityVectors.get(headId);
            ArrayList<Double> labelVector = relationVectors.get(labelId);
            ArrayList<Double> tailVector = entityVectors.get(tailId);
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
     * save the entity vectors in json file in local disk after learning
     * @param entityOutput
     * @throws IOException
     */
    private void writeEntityVector(String entityOutput) throws IOException {
        JSONObject obj = new JSONObject();
        for (int i =0; i< entityVectors.size(); i ++){
            obj.put(entityList.get(i), entityVectors.get(i));
        }

        FileWriter file = new FileWriter(entityOutput);
        file.write(obj.toJSONString());
    }

    /**
     * save the relation vectors in json file in local disk after learning
     * @param relationOutput
     * @throws IOException
     */
    private void writeRelationVector(String relationOutput) throws IOException{
        JSONObject obj = new JSONObject();
        for(int i = 0; i < relationVectors.size(); i ++){
            obj.put(relationList.get(i), relationVectors.get(i));
        }

        FileWriter file = new FileWriter(relationOutput);
        file.write(obj.toJSONString());
    }




}
