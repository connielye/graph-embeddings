package com.prime.common.embedding;

import com.prime.common.computinghelper.Helper;
import com.prime.common.computinghelper.NegativeSampling;
import com.prime.common.io.SerializeModelVectors;
import com.prime.common.io.WriteModel;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;

import java.util.*;

/**
 * TransSparse algorithm -- TransSparse(share) head and tail share the same sparse matrix to project on another space where relation vector is. TransSparse(separate) head and tail have its own correspoinding matrix
 * In this implemention the sparse matrices are unstructured. I cannot think a simple way to generate a structured matrices...
 */

public class TransSparseX {

    final static Logger logger = Logger.getLogger(TransSparseX.class);
    ArrayList<double[]> entityVectors;
    ArrayList<double[]> relationVectors;
    ArrayList<double[][]> matrices;
    ArrayList<double[][]> headMatrices;
    ArrayList<double[][]> tailMatrices;
    Helper helper;
    NegativeSampling sample;
    Boolean separate;

    public TransSparseX(Boolean separate){
        this.helper = new Helper();
        this.sample = new NegativeSampling();
        this.separate = separate;
    }

    /**
     * get the number of none zeros in the sparse matrices minus diagonal numbers for TransSparse(share)
     * @param triples
     * @param relationSize
     * @param theta minimum sparse degree of sparse matrix
     * @param n
     * @return
     */
    private ArrayList<Integer> nonZerosLeft(ArrayList<Triple<Integer, Integer, Integer>> triples, int relationSize, double theta, int n){
        ArrayList<Integer> pairNumbers = new ArrayList<Integer>();
        for (int i =0; i < relationSize; i++){
            int count = 0;
            for (Triple<Integer, Integer, Integer> triple : triples){
                int relation = triple.getMiddle();
                if (relation == i){
                    count ++;
                }
            }
            pairNumbers.add(count);
        }
        ArrayList<Integer> nzLs = new ArrayList<Integer>();
        int max = Collections.max(pairNumbers);
        for (int i = 0; i< pairNumbers.size(); i++) {
            double thetaR = 1 - (1 - theta) * (pairNumbers.get(i) / max);
            int nz = (int) Math.round(thetaR * n * n);
            int nzL;
            if (nz < n) {
                nzL = 0;
            } else {
                nzL = nz - n;
            }
            nzLs.add(nzL);
        }
        return nzLs;

    }

    /**
     * initialize unstructured sparse matrix
      * @param n
     * @param nzL
     * @return
     */
    private double[][] unstructuredSparceMatrix(int n, int nzL){
        double[][] matrix = helper.identityMatrix(n, n);
        int count = 0;
        Random random = new Random();
        while (count < nzL){
            int row = random.nextInt(n);
            int column = random.nextInt(n);
            double number = matrix[row][column];
            if (number == 0){
                double newNumber = helper.initialUnif(n);
                matrix[row][column] = newNumber;
            }
        }
        return matrix;

    }

    /**
     * get the number of none zeros in the sparse matrices minus diagonal numbers for TransSparse(separate)
     * @param triples
     * @param relationSize
     * @param headSet
     * @param theta
     * @param n
     * @return
     */
    private ArrayList<Integer> nonZerosLeftSep(ArrayList<Triple<Integer, Integer, Integer>> triples, int relationSize, Boolean headSet, double theta, int n){
        ArrayList<Integer> entityNumbers = new ArrayList<Integer>();
        for (int i =0; i < relationSize; i++){
            Set<Integer> entitySet = new HashSet<Integer>();
            for (Triple<Integer, Integer, Integer> triple: triples){
                int head = triple.getLeft();
                int relation = triple.getMiddle();
                int tail = triple.getRight();
                if (relation == i){
                    if(headSet == true){
                        entitySet.add(head);
                    }else{
                        entitySet.add(tail);
                    }
                }
            }
            int number = entitySet.size();
            entityNumbers.add(number);
        }
        ArrayList<Integer> nzLs = new ArrayList<Integer>();
        int max = Collections.max(entityNumbers);
        for (int i = 0; i< entityNumbers.size(); i++) {
            double thetaR = 1 - (1 - theta) * (entityNumbers.get(i) / max);
            int nz = (int) Math.round(thetaR * n * n);
            int nzL;
            if (nz < n) {
                nzL = 0;
            } else {
                nzL = nz - n;
            }
            nzLs.add(nzL);
        }
        return nzLs;
    }

    /**
     * initialize entity and relation vectors, and sparse matrices for both models
     * @param transEEntityVectors
     * @param transERelationVectors
     * @param n
     * @param triples
     * @param relationSize
     * @param theta
     */
    private void initialize(ArrayList<double[]> transEEntityVectors, ArrayList<double[]> transERelationVectors, int n, ArrayList<Triple<Integer, Integer, Integer>> triples, int relationSize, double theta){
        entityVectors = transEEntityVectors; // use the result of TransE
        relationVectors = transERelationVectors; // use the result of TransE
        if(separate == true){
            headMatrices = new ArrayList<double[][]>();
            tailMatrices = new ArrayList<double[][]>();
            ArrayList<Integer> headnzL = nonZerosLeftSep(triples, relationSize, true, theta, n);
            for (int h = 0; h < headnzL.size(); h++){
                int nzL = headnzL.get(h);
                double[][] matrix = unstructuredSparceMatrix(n, nzL);
                headMatrices.add(matrix);
            }
            ArrayList<Integer> tailnzL = nonZerosLeftSep(triples, relationSize, false, theta, n);
            for (int t = 0; t < tailnzL.size(); t++){
                int nzL = tailnzL.get(t);
                double[][] matrix = unstructuredSparceMatrix(n, nzL);
                tailMatrices.add(matrix);
            }

        }
        matrices = new ArrayList<double[][]>();
        ArrayList<Integer> nonZeros = nonZerosLeft(triples, relationSize, theta, n);
        for (int i = 0; i < relationSize; i ++){
            int nzL = nonZeros.get(i);
            double[][] matrix = unstructuredSparceMatrix(n, nzL);
            matrices.add(matrix);
        }
    }

    /**
     * sgd
     * @param miniBatch
     * @param negativeTriples
     * @param learningRate
     * @param margin
     * @param n
     * @param L1
     */
    private void update(ArrayList<Triple<Integer, Integer, Integer>> miniBatch, ArrayList<Triple<Integer, Integer, Integer>> negativeTriples, double learningRate, double margin, int n, Boolean L1){
        ArrayList<double[]> copyEntityVectors = (ArrayList<double[]>) entityVectors.clone();
        ArrayList<double[]> copyRelationVectors = (ArrayList<double[]>) relationVectors.clone();
        ArrayList<double[][]> copyMatrices = (ArrayList<double[][]>) matrices.clone();
        ArrayList<double[][]> copyHeadMatrices = (ArrayList<double[][]>) headMatrices.clone();
        ArrayList<double[][]> copyTailMatrices = (ArrayList<double[][]>) tailMatrices.clone();
        for (int i = 0; i < miniBatch.size(); i++){
            Triple<Integer, Integer, Integer> triple = miniBatch.get(i);
            Triple<Integer, Integer, Integer> negativeTriple = negativeTriples.get(i);
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            int headN = negativeTriple.getLeft();
            int tailN = negativeTriple.getRight();
            double[] headNVector = entityVectors.get(headN);
            double[] tailNVector = entityVectors.get(tailN);
            double[] headProjectionVector;
            double[] tailProjectionVector;
            double[] headNProjectionVector;
            double[] tailNProjectionVector;
            double[][] hMatrix = headMatrices.get(relation);
            double[][] tMatrix = tailMatrices.get(relation);
            double[][] matrix = matrices.get(relation);
            if (separate == true){
                headProjectionVector = helper.spaceProjection(headVector, hMatrix);
                headNProjectionVector = helper.spaceProjection(headNVector, hMatrix);
                tailProjectionVector = helper.spaceProjection(tailVector, tMatrix);
                tailNProjectionVector = helper.spaceProjection(tailNVector, tMatrix);
            } else {
                headProjectionVector = helper.spaceProjection(headVector, matrix);
                headNProjectionVector = helper.spaceProjection(headNVector, matrix);
                tailProjectionVector = helper.spaceProjection(tailVector, matrix);
                tailNProjectionVector = helper.spaceProjection(tailNVector, matrix);
            }
            double[] normHeadProjectionVector = helper.norm(headProjectionVector);
            double[] normTailProjectionVector = helper.norm(tailProjectionVector);
            double[] normHeadNProjectionVector = helper.norm(headNProjectionVector);
            double[] normTailNProjectionVector = helper.norm(tailNProjectionVector);
            double distanceL1 = helper.distanceL1(normHeadProjectionVector, relationVector, normTailProjectionVector, n);
            double distanceNL1 = helper.distanceL1(normHeadNProjectionVector, relationVector, normTailNProjectionVector, n);
            double distanceL2 = helper.distanceL2(normHeadProjectionVector, relationVector, normTailProjectionVector, n);
            double distanceNL2 = helper.distanceL2(normHeadNProjectionVector, relationVector, normTailNProjectionVector, n);
            ArrayList<Double> deltaVector = new ArrayList<Double>();
            ArrayList<Double> deltaCVector = new ArrayList<Double>();
            if (L1 == true ){
                double lossL1 = margin + distanceL1 - distanceNL1;
                if (lossL1 > 0){
                    for (int j = 0; j < n; j++){
                        double temp = normTailProjectionVector[j] - normHeadProjectionVector[j] - relationVector[j];
                        double delta;
                        double deltaC;
                        if (temp >= 0){
                            delta = 1d; // first step chain rule
                            deltaVector.add(delta);
                        } else{
                            delta = -1d; // first step chain rule
                            deltaVector.add(delta);
                        }
                        double tempC = normTailNProjectionVector[j] - normHeadNProjectionVector[j] - relationVector[j];
                        if (tempC >= 0){
                            deltaC = 1d; // first step chain rule
                            deltaCVector.add(deltaC);
                        } else {
                            deltaC = 0d; // first step chain rule
                            deltaCVector.add(deltaC);
                        }
                    }
                }
            } else {
                double lossL2 = margin + distanceL2 - distanceNL2;
                if (lossL2 > 0) {
                    for (int m = 0; m < n; m++) {
                        double delta = 2 * (normTailProjectionVector[m] - normHeadProjectionVector[m] - relationVector[m]); // first step chain rule
                        double deltaC = 2 * (normTailNProjectionVector[m] - normHeadNProjectionVector[m] - relationVector[m]); // first step chain rule
                        deltaVector.add(delta);
                        deltaCVector.add(deltaC);
                    }
                }
            }
            if (deltaVector != null) {
                if (separate == true){
                    for (int s = 0; s < n; s++) {
                        double hMatrixDelta = helper.vectorSum(headMatrices.get(relation)[s]);
                        double tMatrixDelta = helper.vectorSum(tailMatrices.get(relation)[s]);
                        double newHead = headVector[s] + learningRate * deltaVector.get(s) * hMatrixDelta; // partial derivation includes matrix, second step chain rule
                        double newRelation = relationVector[s] + learningRate * (deltaVector.get(s) - deltaCVector.get(s)); // relation vector is not affected
                        double newTail = tailVector[s] - learningRate * deltaVector.get(s) * tMatrixDelta; // partial derivation includes matrix, second step chain rule
                        headVector[n] = newHead;
                        relationVector[n] = newRelation;
                        tailVector[n] = newTail;
                        for(int m =0; m < n; m ++){
                            hMatrix[s][m] = hMatrix[s][m] + learningRate * (deltaVector.get(s) *(headVector[s] - tailVector[s]) - deltaVector.get(s) * headNProjectionVector[s]); // update matrix
                            tMatrix[s][m] = tMatrix[s][m] - learningRate * (deltaVector.get(s) *(headVector[s] - tailVector[s]) - deltaVector.get(s) * tailNProjectionVector[s]); // update matrix
                        }
                    }
                    double[] normedHeadVector = helper.norm(headVector);
                    double[] normedLabelVector = helper.norm(relationVector);
                    double[] normedTailVector = helper.norm(tailVector);
                    double[][] normedHMatrix = helper.normMatrix(hMatrix);
                    double[][] normedTMatrix = helper.normMatrix(tMatrix);
                    copyEntityVectors.set(head, normedHeadVector);
                    copyRelationVectors.set(relation, normedLabelVector);
                    copyEntityVectors.set(tail, normedTailVector);
                    copyHeadMatrices.set(relation, normedHMatrix);
                    copyTailMatrices.set(relation, normedTMatrix);
                }
                for (int d = 0; d < n; d++) {
                    double matrixDelta = helper.vectorSum(matrix[d]);
                    double newHead = headVector[d] + learningRate * deltaVector.get(d) * matrixDelta; // partial derivation includes matrix, second step chain rule
                    double newRelation = relationVector[d] + learningRate * (deltaVector.get(d) - deltaCVector.get(d)); // relation vector is not affected.
                    double newTail = tailVector[d] - learningRate * deltaVector.get(d) * matrixDelta; // partial derivation includes matrix, second step chain rule
                    headVector[n] = newHead;
                    relationVector[n] = newRelation;
                    tailVector[n] = newTail;
                    for(int m =0; m < d; m ++){
                        matrix[d][m] = matrix[d][m] + learningRate * (deltaVector.get(d) *(headVector[d] - tailVector[d]) - deltaVector.get(d) *(headNProjectionVector[d] - tailNProjectionVector[d])); // update matrix
                    }
                }
                double[] normedHeadVector = helper.norm(headVector);
                double[] normedLabelVector = helper.norm(relationVector);
                double[] normedTailVector = helper.norm(tailVector);
                double[][] normedMatrix = helper.normMatrix(matrix);
                copyEntityVectors.set(head, normedHeadVector);
                copyRelationVectors.set(relation, normedLabelVector);
                copyEntityVectors.set(tail, normedTailVector);
                copyMatrices.set(relation, normedMatrix);

            }
        entityVectors = copyEntityVectors;
        relationVectors = copyRelationVectors;
        matrices = copyMatrices;
        headMatrices = copyHeadMatrices;
        tailMatrices = copyTailMatrices;
        }
    }

    /**
     * validation on dev set
     * @param devTriples
     * @param margin
     * @param n
     * @param L1
     * @return
     */
    private double validation(ArrayList<Triple<Integer, Integer, Integer>> devTriples, double margin, int n, Boolean L1){
        int count = 0;
        for (Triple<Integer, Integer, Integer> triple : devTriples){
            int head = triple.getLeft();
            int relation = triple.getMiddle();
            int tail = triple.getRight();
            double[] headVector = entityVectors.get(head);
            double[] relationVector = relationVectors.get(relation);
            double[] tailVector = entityVectors.get(tail);
            double[] projectHead;
            double[] projectTail;
            if(separate == true){
                double[][] hMatrix = headMatrices.get(relation);
                projectHead = helper.spaceProjection(headVector, hMatrix);
                double[][] tMatrix = tailMatrices.get(relation);
                projectTail = helper.spaceProjection(tailVector, tMatrix);
            } else {
                double[][] matrix = matrices.get(relation);
                projectHead = helper.spaceProjection(headVector, matrix);
                projectTail = helper.spaceProjection(tailVector, matrix);
            }
            double distance;
            if (L1 == true){
                distance = helper.distanceL1(projectHead, relationVector, projectTail, n);
            } else{
                distance = helper.distanceL2(projectHead, relationVector, projectTail, n);
            }

            if (distance < margin){
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
            entityMap.put(relationName, relationVectors.get(relationId));
        }

        SerializeModelVectors modelEntity = new SerializeModelVectors(entityMap);
        WriteModel writer = new WriteModel();
        writer.write(entityOutput, modelEntity);

        SerializeModelVectors modelRelation = new SerializeModelVectors(relationMap);
        writer.write(relationOutput, modelRelation);

    }

    /**
     * learning phase and cross validation
     * @param trainTriples
     * @param devTriples
     * @param entitySize
     * @param relationSize
     * @param transEEntityVectors
     * @param transRelationVectors
     * @param entity2Id
     * @param relation2Id
     * @param margin
     * @param learningRate
     * @param theta
     * @param n
     * @param size
     * @param epochs
     * @param L1
     * @param entityOutput
     * @param relationOutput
     */
    public void learn(ArrayList<Triple<Integer, Integer, Integer>> trainTriples, ArrayList<Triple<Integer, Integer, Integer>> devTriples, int entitySize, int relationSize, ArrayList<double[]> transEEntityVectors,
                      ArrayList<double[]> transRelationVectors, Map<String, Integer> entity2Id, Map<String, Integer> relation2Id, double margin, double learningRate, double theta, int n, int size, int epochs, Boolean L1,
                      String entityOutput, String relationOutput){
        initialize(transEEntityVectors, transRelationVectors, n, trainTriples, relationSize, theta);
        int batchNumber = Math.round(trainTriples.size() / size);
        for(int i = 0; i < epochs; i++){
            for (int j = 0; j< batchNumber; j ++){
                ArrayList<Triple<Integer, Integer, Integer>> miniBatch = helper.sample(trainTriples, size);
                ArrayList<Triple<Integer, Integer, Integer>> corruptedTriples = sample.generateBern(miniBatch, entitySize, trainTriples, relationSize);
                update(miniBatch, corruptedTriples, learningRate, margin, n, L1);
            }
            logger.info("Epoch: " + i);

            if (devTriples != null){
                double accuracy = validation(devTriples, margin, n, L1);
                logger.info("Validation Accuracy: " + accuracy + "; Epoch: " + i);
            }

        }
        saveModel(entityOutput, relationOutput, entity2Id, relation2Id);

    }
}
