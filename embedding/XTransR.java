package com.prime.common.embedding;

import com.prime.common.computinghelper.Helper;
import com.prime.common.computinghelper.NegativeSampling;
import com.prime.common.computinghelper.RelationCluster;
import com.prime.common.io.SerializeModelLists;
import com.prime.common.io.SerializeModelVectors;
import com.prime.common.io.WriteModel;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * TransR algorithm -- project entity vectors to another space in which relation vectors are. CTransR first uses clustering to get cluster-specific relations and then run sgd similar as TransR
 * Both initialize entity vectors and relation vectors with the result of TransE
 * TransR is complicated and expensive and CTransR is crazy !!!!
 */
public class XTransR {

    final static Logger logger = Logger.getLogger(XTransR.class);
    Helper helper;
    NegativeSampling sample;
    ArrayList<double[]> entityVectors;
    ArrayList<double[]> relationVectors;
    ArrayList<double[][]> matrices;
    Map<Triple<Integer, Integer, Integer>, double[]> relationCVectors;
    Map<double[], double[][]> matricesC;
    Map<Integer, ArrayList<double[]>> relationCluster;
    Boolean CTransR;

    public XTransR(Boolean CTransR){
        this.helper = new Helper();
        this.sample = new NegativeSampling();
        this.CTransR = CTransR;
    }

    /**
     * initialize transR required vectors and matrices
     * @param transEEntityVectors
     * @param transERelationVectors
     * @param k
     * @param d
     */
    private void initialize(ArrayList<double[]> transEEntityVectors, ArrayList<double[]> transERelationVectors, int k, int d){
        entityVectors = transEEntityVectors; // initialize entity embeddings with results from TransE
        relationVectors = transERelationVectors; // initialize relation embeddings with results from TransE
        for(int i =0; i < relationVectors.size(); i++){
            double[][] matrix = helper.identityMatrix(k, d);
            matrices.add(matrix);
        }
    }

    /**
     * using clustering to initialize CTransR required vectors and matrices.
     * @param transEEntityVectors
     * @param transERelationVectors
     * @param relationSize
     * @param triples
     * @param clusterNumber
     * @param clusterEpochs
     * @param k
     * @param d
     */
    private void initializeC(ArrayList<double[]> transEEntityVectors, ArrayList<double[]> transERelationVectors, int relationSize, ArrayList<Triple<Integer, Integer, Integer>> triples, int clusterNumber, int clusterEpochs, int k, int d){
        RelationCluster cluster = new RelationCluster(transEEntityVectors, relationSize, triples, clusterNumber, clusterEpochs, k, d);
        relationCVectors = cluster.getRelationCVectors(); // initialize relation embeddings with results from TransE
        matricesC = cluster.getMatricesMap();
        entityVectors = transEEntityVectors; //initialize entity embeddings with results from TransE
        relationVectors = transERelationVectors;
        relationCluster = cluster.getRelationCluster();
    }

    /**
     * sgd for transR
     * @param miniBatch
     * @param negativeTriples
     * @param learningRate
     * @param k
     * @param d
     * @param margin
     */
    private void update(ArrayList<Triple<Integer, Integer, Integer>>miniBatch, ArrayList<Triple<Integer, Integer, Integer>> negativeTriples, double learningRate, int k, int d, double margin){
        ArrayList<double[]> copyEntityVectors = (ArrayList<double[]>) entityVectors.clone();
        ArrayList<double[]> copyRelationVectors = (ArrayList<double[]>) relationVectors.clone();
        ArrayList<double[][]> copyMatrices = (ArrayList<double[][]>) matrices.clone();
        for (int i = 0; i < miniBatch.size(); i++){
            Triple<Integer, Integer, Integer> triple = miniBatch.get(i);
            Triple<Integer, Integer, Integer> negativeTriple = negativeTriples.get(i);
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            double[][] matrix = matrices.get(relation);
            double[] headRVector = helper.spaceProjection(headVector, matrix); // project vector to another space
            double[] normHeadRVector = helper.norm(headRVector);
            double[] tailRVector = helper.spaceProjection(tailVector, matrix); // project vector to another space
            double[] normTailRVector = helper.norm(tailRVector);
            double distanceL2 = helper.distanceL2(normHeadRVector, relationVector, normTailRVector, k);
            int headN = negativeTriple.getLeft();
            int tailN = negativeTriple.getRight();
            double[] headNVector = entityVectors.get(headN);
            double[] tailNVector = entityVectors.get(tailN);
            double[] headNRVector = helper.spaceProjection(headNVector, matrix); // project vector to another space
            double[] normHeadNRVector = helper.norm(headNRVector);
            double[] tailNRVector = helper.spaceProjection(tailNVector, matrix); // project vector to another space
            double[] normTailNRVector = helper.norm(tailNRVector);
            double distanceNL2 = helper.distanceL2(normHeadNRVector, relationVector, normTailNRVector, k);
            double loss = distanceL2 + margin - distanceNL2;
            if (loss > 0){
                double[] newHead = new double[k];
                double[] newRelation = new double[k];
                double[] newTail = new double[k];
                double[][] newMatrix = new double[k][d];
                for (int j =0; j<k; j++){
                    double delta = 2 *(normTailRVector[j] - normHeadRVector[j] - relationVector[j]); // first step chain rule
                    double deltaC = 2* (normTailNRVector[j] - normHeadNRVector[j] - relationVector[j]); // first step chain rule
                    double matrixDelta = helper.vectorSum(matrix[j]);
                    newHead[j] = headVector[j] + learningRate * delta * matrixDelta; // the partial derivative includes matrix, second step chain rule
                    newTail[j] = tailVector[j] - learningRate * delta * matrixDelta; // the partial derivative includes matrix, second step chain rule
                    newRelation[j] = relationVector[j] + learningRate * (delta - deltaC); // relation vector is not affected
                    for(int m =0; m < d; m ++){
                        newMatrix[j][m] = matrix[j][m] + learningRate * (delta *(headVector[j] - tailVector[j]) - deltaC *(headNRVector[j] - tailNRVector[j])); // update the matrix
                    }
                }
                double[] normNewHead = helper.norm(newHead);
                double[] normNewRelation = helper.norm(newRelation);
                double[] normNewTail = helper.norm(newTail);
                double[][] normNewMatrix = helper.normMatrix(newMatrix);
                copyEntityVectors.set(head, normNewHead);
                copyEntityVectors.set(tail, normNewTail);
                copyRelationVectors.set(relation, normNewRelation);
                copyMatrices.set(relation, normNewMatrix);
            }
        }
        entityVectors = copyEntityVectors;
        relationVectors = copyRelationVectors;
        matrices = copyMatrices;
    }


    /**
     * sgd for CTransR
     * @param miniBatch
     * @param negativeTriples
     * @param learningRate
     * @param k
     * @param d
     * @param margin
     * @param alpha
     */
    private void updateC(ArrayList<Triple<Integer, Integer, Integer>>miniBatch, ArrayList<Triple<Integer, Integer, Integer>> negativeTriples, double learningRate, int k, int d, double margin, double alpha){
        ArrayList<double[]> copyEntityVectors = (ArrayList<double[]>) entityVectors.clone();
        ArrayList<double[]> copyRelationVectors = (ArrayList<double[]>) relationVectors.clone();
        for (int i = 0; i<miniBatch.size(); i++){
            Triple<Integer, Integer,Integer> triple = miniBatch.get(i);
            Triple<Integer, Integer, Integer> negativeTriple = negativeTriples.get(i);
            double[] relationC = relationCVectors.get(triple); // get cluster-specific relation vector
            double[][] matrix = matricesC.get(relationC); // get matrix according to relationC
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            double[] headRVector = helper.spaceProjection(headVector, matrix); // project to another space using relationC corresponding matrix
            double[] normHeadRVector = helper.norm(headRVector);
            double[] tailRVector = helper.spaceProjection(tailVector, matrix); // project to another space using relationC corresponding matrix
            double[] normTailRVector = helper.norm(tailRVector);
            int headN = negativeTriple.getLeft();
            int tailN = negativeTriple.getRight();
            double[] headNVector = entityVectors.get(headN);
            double[] tailNVector = entityVectors.get(tailN);
            double[] headNRVector = helper.spaceProjection(headNVector, matrix); // project to another space using relationC corresponding matrix
            double[] normHeadNRVector = helper.norm(headNRVector);
            double[] tailNRVector = helper.spaceProjection(tailNVector, matrix); // project to another space using relationC corresponding matrix
            double[] normTailNRVector = helper.norm(tailNRVector);
            double relationDistance = helper.relationDistanceL2(relationC, relationVector); // constraint ensure relationC and relation is not far away
            double distance = helper.distanceL2(normHeadRVector, relationC, normTailNRVector, k) + alpha * relationDistance; // distance includes relation distance
            double distanceN = helper.distanceL2(normHeadNRVector, relationC, normTailNRVector, k) + alpha * relationDistance; // distance includes relation distance
            double loss = distance + margin - distanceN;
            if (loss > 0){
                double[] newHead = new double[k];
                double[] newRelation = new double[k];
                double[] newRelationC = new double[k];
                double[] newTail = new double[k];
                double[][] newMatrix = new double[k][d];
                for (int j =0; j<k; j++){
                    double delta = 2 * ((normTailRVector[j] - normHeadRVector[j] - relationVector[j])+ alpha * relationDistance); // partial derivation includes relation distance, first step chain rule
                    double deltaC = 2* ((normTailNRVector[j] - normHeadNRVector[j] - relationVector[j] + alpha * relationDistance)); // partial derivation includes relation distance, first step chain rule
                    double matrixDelta = helper.vectorSum(matrix[j]);
                    newHead[j] = headVector[j] + learningRate * delta * matrixDelta; // partial derivation includes matrix, second step chain rule
                    newTail[j] = tailVector[j] - learningRate * delta * matrixDelta; // partial derivation includes matrix, second step chain rule
                    newRelation[j] = relationVector[j] - learningRate * (delta * alpha - deltaC *alpha); // relation is not affected by matrix but affected by relation distance
                    newRelationC[j] = relationC[j] + learningRate *(delta *(1 + alpha) - deltaC * (1 + alpha)); // relationC is not affected by matrix but affected by relation distance
                    for(int m =0; m < d; m ++){
                        newMatrix[j][m] = matrix[j][m] + learningRate * (delta *(headVector[j] - tailVector[j]) - deltaC *(headNRVector[j] - tailNRVector[j])); // update matrix
                    }
                }
                double[] normNewHead = helper.norm(newHead);
                double[] normNewRelation = helper.norm(newRelation);
                double[] normNewTail = helper.norm(newTail);
                double[] normNewRelationC = helper.norm(newRelationC);
                double[][] normNewMatrix = helper.normMatrix(newMatrix);
                copyEntityVectors.set(head, normNewHead);
                copyEntityVectors.set(tail, normNewTail);
                copyRelationVectors.set(relation, normNewRelation);
                relationCVectors.put(triple, normNewRelationC);
                matricesC.remove(relationC);
                matricesC.put(normNewRelationC, normNewMatrix);
                ArrayList<double[]> relationCs = relationCluster.get(relation);
                for(double [] vector: relationCs){
                    if (vector == relationC){
                        int index = relationCs.indexOf(vector);
                        relationCs.set(index, normNewRelationC);
                    }
                }
                relationCluster.remove(relation);
                relationCluster.put(relation, relationCs); //update relationC list
            }
        }
        entityVectors = copyEntityVectors;
        relationVectors = copyEntityVectors;
    }

    /**
     * validation on dev set for transR
     * @param devTriples
     * @param margin
     * @param k
     * @return
     */

    private double validation(ArrayList<Triple<Integer, Integer, Integer>> devTriples, double margin, int k){
        int count = 0;
        for (Triple<Integer, Integer, Integer> triple : devTriples){
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            double[][] matrix = matrices.get(relation);
            double[] projectHead = helper.spaceProjection(headVector, matrix);
            double[] projectTail = helper.spaceProjection(tailVector, matrix);
            double distanceL2 = helper.distanceL2(projectHead, relationVector, projectTail, k);
            if (distanceL2 < margin){
                count ++;
            }
        }
        double accuracy = count / devTriples.size();
        return accuracy;
    }

    /**
     * validation on dev set for CTransR, because one relation may have k cluster-specific relations. I use euclidean distance between relationC and offset (head - tail) to decide which is the relationC for the pair
     * @param devTriples
     * @param margin
     * @param k
     * @return
     */
    private double validationC(ArrayList<Triple<Integer, Integer, Integer>> devTriples, double margin, int k){
        int count = 0;
        for (Triple<Integer, Integer, Integer> triple : devTriples){
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            ArrayList<double[]> relationCs = relationCluster.get(relation);
            double[] offset = new double[k];
            for (int i = 0; i< headVector.length; i++){
                offset[i] = headVector[i] - tailVector[i];
            }
            double distance = helper.euclideanDistance(offset, relationCs.get(0));
            int index = 0;
            for (int i =1; i < relationCs.size(); i ++){
                double[] vector = relationCs.get(i);
                double edistance = helper.euclideanDistance(offset, vector);
                if (edistance < distance){
                    distance = edistance;
                    index = i;
                }
            }
            double[] relationC = relationCs.get(index);
            double[][] matrix = matricesC.get(relationC);
            double[] projectHead = helper.spaceProjection(headVector, matrix);
            double[] projectTail = helper.spaceProjection(tailVector, matrix);
            double distanceL2 = helper.distanceL2(projectHead, relationVector, projectTail, k);
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
     * @param relationCOutput
     * @param entity2Id
     * @param relation2Id
     */
    private void saveModel(String entityOutput, String relationOutput, String relationCOutput, Map<String, Integer> entity2Id, Map<String, Integer> relation2Id){
        Map<String, double[]> entityMap = new HashMap<String, double[]>();
        Map<String, double[]> relationMap = new HashMap<String, double[]>();
        Map<String, ArrayList<double[]>> relationCMap = new HashMap<String, ArrayList<double[]>>();
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
            entityMap.put(relationName, relationVectors.get(relationId));
            relationCMap.put(relationName, relationCluster.get(relationId));
        }

        SerializeModelVectors modelEntity = new SerializeModelVectors(entityMap);
        WriteModel writer = new WriteModel();
        writer.write(entityOutput, modelEntity);

        SerializeModelVectors modelRelation = new SerializeModelVectors(relationMap);
        writer.write(relationOutput, modelRelation);

        SerializeModelLists modelRelationC = new SerializeModelLists(relationCMap);
        writer.write2(relationCOutput, modelRelationC);


    }


    /**
     * learning phase and cross validation after each epoch
     * @param trainTriples
     * @param devTriples
     * @param transEEntityVectors
     * @param transERelationVectors
     * @param relationSize
     * @param enitytSize
     * @param learningRate
     * @param margin
     * @param k
     * @param d
     * @param alpha
     * @param clusterNumber
     * @param clusterEpochs
     * @param epochs
     * @param size
     * @param entityOutput
     * @param relationOutput
     * @param relationCOutput
     * @param entity2Id
     * @param relation2Id
     */
    public void learn (ArrayList<Triple<Integer, Integer, Integer>> trainTriples, ArrayList<Triple<Integer, Integer, Integer>> devTriples, ArrayList<double[]> transEEntityVectors, ArrayList<double[]> transERelationVectors, int relationSize,
                       int enitytSize, double learningRate, double margin, int k, int d, double alpha, int clusterNumber, int clusterEpochs, int epochs, int size, String entityOutput, String relationOutput, String relationCOutput,
                       Map<String, Integer> entity2Id, Map<String, Integer> relation2Id){
        if(CTransR){
            initializeC(transEEntityVectors, transERelationVectors, relationSize, trainTriples, clusterNumber, clusterEpochs, k, d);
        } else {
            initialize(transEEntityVectors, transERelationVectors, k, d);
        }
        int batchNumber = Math.round(trainTriples.size() / size);
        for(int i = 0; i < epochs; i++){
            for (int j = 0; j< batchNumber; j ++) {
                ArrayList<Triple<Integer, Integer, Integer>> miniBatch = helper.sample(trainTriples, size);
                ArrayList<Triple<Integer, Integer, Integer>> corruptedTriples = sample.generateBern(miniBatch, enitytSize, trainTriples, relationSize);
                if(CTransR){
                    updateC(miniBatch, corruptedTriples, learningRate, k, d, margin, alpha);
                } else{
                    update(miniBatch, corruptedTriples, learningRate, k, d, margin);
                }
            }
            logger.info("Epoch: " + i);

            if(devTriples != null){
                double accuracy = 0d;
                if(CTransR){
                    accuracy = validationC(devTriples, margin, k);
                } else {
                    accuracy = validation(devTriples, margin, k);
                }
                logger.info("Validation Accuracy: " + accuracy + "; Epoch: " + i);
            }

        }
        saveModel(entityOutput, relationOutput, relationCOutput, entity2Id, relation2Id);


    }




}
