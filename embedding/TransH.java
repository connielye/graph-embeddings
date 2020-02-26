package com.prime.common.embedding;

import com.prime.common.computinghelper.Helper;
import com.prime.common.computinghelper.NegativeSampling;
import com.prime.common.io.SerializeModelVectors;
import com.prime.common.io.WriteModel;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;

import java.util.*;

/**
 * TransH algorithm (projecting vector to a plane where relation vector is) -- reflects one-many/ many-one/ many-many relations
 */


public class TransH {

    final static Logger logger = Logger.getLogger(TransH.class);
    Helper helper;
    NegativeSampling sample;
    ArrayList<double[]> entityVectors;
    ArrayList<double[]> relationVectors;
    ArrayList<double[]> normalVectors;

    public TransH(){
        this.helper = new Helper();
        this.sample = new NegativeSampling();
    }

    /**
     * initialize entity vectors, relation vectors and normal vectors
     * @param k
     */

    private void initialize(int entitySize, int relationSize,int k){
        entityVectors = new ArrayList<double[]>();
        relationVectors = new ArrayList<double[]>();
        normalVectors = new ArrayList<double[]>();
        for (int i = 0; i < entitySize; i++){
            double[] eVector = helper.initVector(k);
            double[] normedEVector = helper.norm(eVector);
            entityVectors.add(normedEVector);
        }
        for(int j =0; j < relationSize; j++){
            double[] rVector = helper.initVector(k);
            double[] normRVector = helper.norm(rVector);
            double[] nVector = helper.initVector(k);
            double[]normNVector = helper.norm(nVector);
            relationVectors.add(normRVector);
            normalVectors.add(normNVector);
        }
    }



    /**
     * sgd, the only different is to use project vector of the cost function instead of entity vectors in TransE and there is a soft constraint on orthogonality
     * @param miniBatch
     * @param corruptedTriples
     * @param margin
     * @param learningRate
     * @param C hyper-parameter weighting the importance of soft constraints
     * @param k
     */

    private void update(ArrayList<Triple<Integer, Integer, Integer>> miniBatch, ArrayList<Triple<Integer, Integer, Integer>> corruptedTriples, double margin, double learningRate, double C, int k){
        ArrayList<double[]> copyEntityVectors = (ArrayList<double[]>) entityVectors.clone();
        ArrayList<double[]> copyRelationVectors = (ArrayList<double[]>) relationVectors.clone();
        ArrayList<double[]> copyNormalVectors = (ArrayList<double[]>) normalVectors.clone();

        for(int i = 0; i < miniBatch.size(); i ++){
            int head = miniBatch.get(i).getLeft();
            int relation = miniBatch.get(i).getMiddle();
            int tail = miniBatch.get(i).getRight();
            int headC = corruptedTriples.get(i).getLeft();
            int tailC = corruptedTriples.get(i).getRight();
            double[] normalVector = copyNormalVectors.get(relation);
            double[] headVector = copyEntityVectors.get(head);
            double[] relationVector = copyRelationVectors.get(relation);
            double[] tailVector = copyEntityVectors.get(tail);
            double[] headCVector = copyEntityVectors.get(headC);
            double[] tailCVector = copyEntityVectors.get(tailC);
            double[] projectHead = helper.planeProjection(headVector, normalVector); // project vector to a plane
            double[] projectTail = helper.planeProjection(tailVector, normalVector); // project vector to a plane
            double[] projectHeadC = helper.planeProjection(headCVector, normalVector); // project vector to a plane
            double[] projectTailC = helper.planeProjection(tailCVector, normalVector); // project vector to a plane

            Boolean orthogonal = helper.checkOrthogonal(relationVector, normalVector);
            double dotProduct = helper.dotProduct(relationVector, normalVector);

            double distanceL2 = helper.distanceL2(projectHead, relationVector, projectTail, k);
            double distanceL2C = helper.distanceL2(projectHeadC, relationVector, projectTailC, k);
            double loss = distanceL2 + margin - distanceL2C;
            if (loss > 0){
                double[] newHeadVector = new double[k];
                double[] newRelationVector = new double[k];
                double[] newTailVector = new double[k];
                double[]newNormalVector = new double[k];
                for (int j = 0; j< k; j++){
                    double delta = 2*(projectTail[j] - projectHead[j] -relationVector[j]);
                    double deltaC = 2*(projectTailC[j] - projectHeadC[j] - relationVector[j]);
                    double constraintR = 0;
                    double constraintN = 0;
                    if (orthogonal == false){
                        constraintR = 2 *C * dotProduct *normalVector[j]; // constraints if the relation vector and the normal vector is not orthogonal for relation vector
                        constraintN = 2* C * dotProduct *relationVector[j]; // constraints if not orthogonal for normal vector
                    }
                    double newHead = headVector[j] + learningRate * delta * (1 - Math.pow(normalVector[j], 2)); // the partial derivative includes normal vector, second step chain rule
                    double newRelation = relationVector[j] + learningRate * (delta - deltaC) - constraintR; // relation vector is not affected by normal vector, second step chain rule
                    double newTail = tailVector[j] - learningRate * delta * (1 - Math.pow(normalVector[j], 2)); // the partial derivative includes normal vector
                    double newNormal = normalVector[j] - constraintN; // update normal vector if not orthogonal
                    newHeadVector[j] = newHead;
                    newRelationVector[j] = newRelation;
                    newTailVector[j] = newTail;
                    newNormalVector[j] = newNormal;
                }
                double[] normedHeadVector = helper.norm(newHeadVector);
                double[] normedRelationVector = helper.norm(newRelationVector);
                double[] normedTailVector = helper.norm(newTailVector);
                double[] normedNormalVector = helper.norm(newNormalVector);
                copyEntityVectors.set(head, normedHeadVector);
                copyEntityVectors.set(tail, normedTailVector);
                copyRelationVectors.set(relation, normedRelationVector);
                copyNormalVectors.set(relation, normedNormalVector);
            }
        }
        entityVectors = copyEntityVectors;
        relationVectors = copyRelationVectors;
        normalVectors = copyNormalVectors;
    }

    /**
     * validation on dev set
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
            double[] normalVector = normalVectors.get(relation);
            double[] projectHead = helper.planeProjection(headVector, normalVector);
            double[] projectTail = helper.planeProjection(tailVector, normalVector);
            double distanceL2 = helper.distanceL2(projectHead, relationVector, projectTail, k);
            if (distanceL2 < margin){
                count ++;
            }
        }
        double accuracy = count / devTriples.size();
        return accuracy;
    }

    /**
     * save trained model in a local disk
     * @param entityOutput
     * @param relationOutput
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
     * learning phase and cross validation after each epoch
     * @param devTriples
     * @param margin
     * @param learningRate
     * @param C
     * @param k
     * @param size
     * @param epochs
     * @param entityOutput
     * @param relationOutput
     */

    public void learn(ArrayList<Triple<Integer, Integer, Integer>> trainTriples, ArrayList<Triple<Integer, Integer, Integer>> devTriples, int entitySize, int relationSize, Map<String, Integer> entity2Id,
                      Map<String, Integer> relation2Id, double margin, double learningRate, double C, int k, int size, int epochs, String entityOutput, String relationOutput){
        initialize(entitySize, relationSize, k);
        int batchNumber = Math.round(trainTriples.size() / size);
        for(int i = 0; i < epochs; i++){
            for (int j = 0; j< batchNumber; j ++){
                ArrayList<Triple<Integer, Integer, Integer>> miniBatch = helper.sample(trainTriples, size);
                ArrayList<Triple<Integer, Integer, Integer>> corruptedTriples = sample.generateBern(miniBatch, entitySize, trainTriples, relationSize);
                update(miniBatch, corruptedTriples, margin, learningRate, C, k);
            }
            logger.info("Epoch: " + i);

            if (devTriples != null){
                double accuracy = validation(devTriples, margin, k);
                logger.info("Validation Accuracy: " + accuracy + "; Epoch: " + i);
            }

        }
        saveModel(entityOutput, relationOutput, entity2Id, relation2Id);

    }










}
