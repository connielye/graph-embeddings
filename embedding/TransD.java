package com.prime.common.embedding;


import com.prime.common.computinghelper.Helper;
import com.prime.common.computinghelper.NegativeSampling;
import com.prime.common.io.Node2Id;
import com.prime.common.io.ReadText;
import com.prime.common.io.SerializeModelVectors;
import com.prime.common.io.WriteModel;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * TransD algorithm -- project vectors to another space where relation are using vector computing (actually easy computing from identity matrix) instead of matrix. Each entity pair (head, tail) has its specific mapping matrix
 * TransD's computing is easier compared to transR and, citing the paper, "can be applied to large scale graphs".
 */
public class TransD {

    final static Logger logger = Logger.getLogger(TransD.class);
    Helper helper;
    NegativeSampling sample;
    ArrayList<double[]> entityVectors;
    ArrayList<double[]> entityProjectionVectors;
    ArrayList<double[]> relationVectors;
    ArrayList<double[]> relationProjectVectors;

    public TransD(){
        this.helper = new Helper();
        this.sample = new NegativeSampling();
    }

    /**
     * initialize entity vectors, relation vectors and their projection vectors. The paper assume m >= n.
     * @param entitySize
     * @param n this is dimension of entity vector
     * @param relationSize
     * @param m this is dimension of relation vector
     */

    private void initialize(int entitySize, int n, int relationSize, int m){
        entityVectors = new ArrayList<double[]>();
        entityProjectionVectors = new ArrayList<double[]>();
        relationVectors = new ArrayList<double[]>();
        relationProjectVectors = new ArrayList<double[]>();
        for (int i = 0; i<entitySize; i++){
            double[] eVector= helper.initVector(n);
            double[] normedEVector = helper.norm(eVector);
            entityVectors.add(normedEVector);
            entityProjectionVectors.add(normedEVector);
        }
        for (int j =0; j<relationSize; j++){
            double[] rVector= helper.initVector(m);
            double[] normedRVector = helper.norm(rVector);
            relationVectors.add(normedRVector);
            relationProjectVectors.add(normedRVector);
        }
    }

    /**
     * project vector to another space by vector computing. The original equation is using identity matrix which can instead compute as a vector.
     * @param entityVector
     * @param entityProjectionVector
     * @param relationProjectionVector
     * @return projected vector with dimension of m (relation vector dimension)
     */

    private double[] entityProjectedVector(double[] entityVector, double[] entityProjectionVector, double[] relationProjectionVector){
        double[] eProjectedVector = new double[entityVector.length];
        double dotProduct = helper.dotProduct(entityVector, entityProjectionVector);
        for (int i =0; i<relationProjectionVector.length; i++){
            double number;
            if(entityVector.length < i){
                number = dotProduct * relationProjectionVector[i]; // in case m > n.
            } else{
                number = dotProduct*relationProjectionVector[i] + entityVector[i];
            }
            eProjectedVector[i] = number;
        }
        double[] normedVector = helper.norm(eProjectedVector);
        return normedVector;
    }

    /**
     * sgd  note that m and n may not be equal.
     * @param miniBatch
     * @param negativeTriples
     * @param learningRate
     * @param margin
     * @param n
     * @param m
     */

    private void update (ArrayList<Triple<Integer, Integer, Integer>> miniBatch, ArrayList<Triple<Integer, Integer, Integer>> negativeTriples, double learningRate, double margin, int n, int m){
        ArrayList<double[]> copyEntityVectors = (ArrayList<double[]>) entityVectors.clone();
        ArrayList<double[]> copyEntityProjectionVectors = (ArrayList<double[]>) entityProjectionVectors.clone();
        ArrayList<double[]> copyRelationVectors = (ArrayList<double[]>) relationVectors.clone();
        ArrayList<double[]> copyRelationProjectionVectors = (ArrayList<double[]>) relationProjectVectors.clone();
        for (int i =0; i<miniBatch.size(); i++){
            Triple<Integer, Integer, Integer> triple = miniBatch.get(i);
            Triple<Integer, Integer, Integer> negativeTriple = negativeTriples.get(i);
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            int headC = negativeTriple.getLeft();
            int tailC = negativeTriple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] headProjectionVector = entityProjectionVectors.get(head);
            double[] tailVector= entityVectors.get(tail);
            double[] tailProjectionVector = entityProjectionVectors.get(tail);
            double[] relationVector = relationVectors.get(relation);
            double[] relationProjectionVector = relationProjectVectors.get(relation);
            double[] headCVector = entityVectors.get(headC);
            double[] headCProjectionVector = entityProjectionVectors.get(headC);
            double[] tailCVector = entityVectors.get(tailC);
            double[] tailCProjectionVector = entityProjectionVectors.get(tailC);
            double[] headProjectedVector = entityProjectedVector(headVector, headProjectionVector, relationProjectionVector);
            double[] tailProjectedVector = entityProjectedVector(tailVector, tailProjectionVector, relationProjectionVector);
            double[] headCProjectedVector = entityProjectedVector(headCVector, headCProjectionVector, relationProjectionVector);
            double[] tailCProjectedVector = entityProjectedVector(tailCVector, tailCProjectionVector, relationProjectionVector);
            double distanceL2 = helper.distanceL2(headProjectedVector, relationVector, tailProjectedVector, n);
            double distanceCL2 = helper.distanceL2(headCProjectedVector, relationVector, tailCProjectedVector, n);
            double loss = margin + distanceL2 - distanceCL2 ;
            if (loss > 0){
                double[] newHead = new double[n];
                double[] newProjectionHead = new double[n];
                double[] newTail = new double[n];
                double[] newProjectionTail = new double[n];
                double[] newRelation = new double[m];
                double[] newProjectionRelation = new double[m];
                for (int j =0; j<n;j++){ // update entity vectors
                    double delta = 2*(tailProjectedVector[j] - headProjectedVector[j] - relationVector[j]); // first step chain rule
                    newHead[j] = headVector[j] + learningRate * delta * (headProjectionVector[j] * relationProjectionVector[j]); // the partial derivative includes projection, second step chain rule
                    newProjectionHead[j] = headProjectionVector[j] + learningRate * delta * (headVector[j] * relationProjectionVector[j]); // the partial derivative includes projection, second step chain rule
                    newTail[j] = tailVector[j] - learningRate * delta * (tailProjectedVector[j] * relationProjectionVector[j]); // the partial derivative includes projection, second step chain rule
                    newProjectionTail[j] = tailProjectedVector[j] - learningRate * delta * (tailVector[j] * relationProjectionVector[j]); // the partial derivative includes projection, second step chain rule
                }
                for (int s = 0; s<m; s++){ // update relation vectors
                    double delta = 2*(tailProjectedVector[s] - headProjectedVector[s] - relationVector[s]); // first step chain rule
                    double deltaC = 2*(tailCProjectedVector[s] - headCProjectedVector[s] - relationVector[s]); // first step chain rule
                    newRelation[s] = relationVector[s] + learningRate * (delta - deltaC); // relation vector is not affected by projection
                    double dotProductH = helper.dotProduct(headVector, headProjectionVector);
                    double dotProductT = helper.dotProduct(tailVector, tailProjectionVector);
                    newProjectionRelation[s] = relationProjectionVector[s] + learningRate * (delta - deltaC) * (dotProductH - dotProductT); // the partial derivative includes projection, second step chain rule
                }
                double[] normedNewHead = helper.norm(newHead);
                double[] normedNewProjectionHead = helper.norm(newProjectionHead);
                double[] normedNewTail = helper.norm(newTail);
                double[] normedNewProjectionTail = helper.norm(newProjectionTail);
                double[] normedNewRelation = helper.norm(newRelation);
                double[] normedNewProjectionRelation = helper.norm(newProjectionRelation);
                copyEntityVectors.set(head, normedNewHead);
                copyEntityProjectionVectors.set(head, normedNewProjectionHead);
                copyEntityVectors.set(tail, normedNewTail);
                copyEntityProjectionVectors.set(tail, normedNewProjectionTail);
                copyRelationVectors.set(relation, normedNewRelation);
                copyRelationProjectionVectors.set(relation, normedNewProjectionRelation);
            }

        }
        entityVectors = copyEntityVectors;
        entityProjectionVectors = copyEntityProjectionVectors;
        relationVectors = copyRelationVectors;
        relationProjectVectors = copyRelationProjectionVectors;
    }

    /**
     * validation on dev set for transD
     * @param devTriples
     * @param margin
     * @param m
     * @return
     */

    private double validation(ArrayList<Triple<Integer, Integer, Integer>> devTriples, double margin, int m){
        int count = 0;
        for (Triple<Integer, Integer, Integer> triple : devTriples){
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] headProjectionVector = entityProjectionVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] relationProjectionVector = relationProjectVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            double[] tailProjectionVector = entityProjectionVectors.get(tail);
            double[] projectedHead = entityProjectedVector(headVector, headProjectionVector, relationProjectionVector);
            double[] projectedTail = entityProjectedVector(tailVector, tailProjectionVector, relationProjectionVector);
            double distanceL2 = helper.distanceL2(projectedHead, relationVector, projectedTail, m);
            if (distanceL2 < margin){
                count ++;
            }
        }
        double accuracy = count / devTriples.size();
        return accuracy;
    }

    /**
     * save trained model in local disk
     * @param entityOutput
     * @param relationOutput
     * @param entity2Id
     * @param relation2Id
     */
    private void saveModel(String entityOutput, String relationOutput, Map<String, Integer> entity2Id, Map<String, Integer> relation2Id){
        Map<String, double[]> entityMap = new HashMap<String, double[]>();
        Map<String, double[]> entityProjectionMap = new HashMap<String, double[]>();
        Map<String, double[]> relationMap = new HashMap<String, double[]>();
        Map<String, double[]> relationProjectionMap = new HashMap<String, double[]>();
        Iterator iteratorEntity = entity2Id.entrySet().iterator();
        while (iteratorEntity.hasNext()){
            Map.Entry entry = (Map.Entry)iteratorEntity.next();
            String entityName = entry.getKey().toString();
            int entityId = Integer.parseInt(entry.getValue().toString());
            entityMap.put(entityName, entityVectors.get(entityId));
        }
        while (iteratorEntity.hasNext()){
            Map.Entry entry = (Map.Entry)iteratorEntity.next();
            String entityName = entry.getKey().toString();
            int entityId = Integer.parseInt(entry.getValue().toString());
            entityProjectionMap.put(entityName, entityProjectionVectors.get(entityId));
        }
        Iterator iteratorRelation = relation2Id.entrySet().iterator();
        while (iteratorRelation.hasNext()){
            Map.Entry entry = (Map.Entry)iteratorRelation.next();
            String relationName = entry.getKey().toString();
            int relationId = Integer.parseInt(entry.getValue().toString());
            entityMap.put(relationName, relationVectors.get(relationId));
        }
        while (iteratorRelation.hasNext()){
            Map.Entry entry = (Map.Entry)iteratorRelation.next();
            String relationName = entry.getKey().toString();
            int relationId = Integer.parseInt(entry.getValue().toString());
            entityProjectionMap.put(relationName, relationProjectVectors.get(relationId));
        }


        SerializeModelVectors modelEntity = new SerializeModelVectors(entityMap);
        WriteModel writer = new WriteModel();
        writer.write(entityOutput, modelEntity);
        SerializeModelVectors modelEntityProjection = new SerializeModelVectors(entityProjectionMap);
        writer.write(relationOutput, modelEntityProjection);

        SerializeModelVectors modelRelation = new SerializeModelVectors(relationMap);
        writer.write(relationOutput, modelRelation);
        SerializeModelVectors modelRelationProjection = new SerializeModelVectors(relationProjectionMap);
        writer.write(relationOutput, modelRelationProjection);


    }

    /**
     * learning phase and cross validation on dev set
     * @param trainTriples
     * @param devTriples
     * @param entitySize
     * @param relationSize
     * @param entity2Id
     * @param relation2Id
     * @param margin
     * @param learningRate
     * @param n
     * @param m
     * @param size
     * @param epochs
     * @param entityOutput
     * @param relationOutput
     */
    public void learn(ArrayList<Triple<Integer, Integer, Integer>> trainTriples, ArrayList<Triple<Integer, Integer, Integer>> devTriples, int entitySize, int relationSize, Map<String, Integer> entity2Id,
                      Map<String, Integer> relation2Id, double margin, double learningRate, int n, int m, int size, int epochs, String entityOutput, String relationOutput){
        initialize(entitySize, n,relationSize, m);
        int batchNumber = Math.round(trainTriples.size() / size);
        for(int i = 0; i < epochs; i++){
            for (int j = 0; j< batchNumber; j ++){
                ArrayList<Triple<Integer, Integer, Integer>> miniBatch = helper.sample(trainTriples, size);
                ArrayList<Triple<Integer, Integer, Integer>> corruptedTriples = sample.generateBern(miniBatch, entitySize, trainTriples, relationSize);
                update(miniBatch, corruptedTriples, learningRate, margin, n, m);
            }
            logger.info("Epoch: " + i);

            if (devTriples != null){
                double accuracy = validation(devTriples, margin, n);
                logger.info("Validation Accuracy: " + accuracy + "; Epoch: " + i);
            }

        }
        saveModel(entityOutput, relationOutput, entity2Id, relation2Id);

    }

}
