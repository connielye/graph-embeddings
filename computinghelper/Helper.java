package com.prime.common.computinghelper;

import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.SingularValueDecomposition;


import java.util.ArrayList;
import java.util.Random;

public class Helper {

    final double epsilon = 1e-12;
   /**
     * initialize vectors from (-6/sqr k, 6/sqr k) within uniform distribution
     * @param k for com.prime.common.embedding dimensions
     * @return initialized vector
     */
    public double[] initVector(int k){
        double[] vector = new double[k];
        for (int i = 0; i < k; i++) {
            double low = -(6 / Math.sqrt(k));
            double high = 6 / Math.sqrt(k);
            double random = new UniformRealDistribution(low, high).sample();
            vector[i] = random;
        }
        return vector;

    }

    public double initialUnif(int k){
        double low = -(6 / Math.sqrt(k));
        double high = 6 / Math.sqrt(k);
        double random = new UniformRealDistribution(low, high).sample();
        return random;
    }

    /**
     * normlize vectors the norm of entity and relation is 1
     * @param vector
     * @return normlized vectors
     */
    public double[] norm(double[] vector){
        double[] result = new double[vector.length];
        double sum = 0;
        for (double number: vector){
            sum = sum + Math.pow(number, 2d);
        }

        double denominator = Math.sqrt(sum);

        for (int i = 0; i < vector.length; i++){
            double normNumber = vector[i] / denominator;
            result[i] = normNumber;
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

    public double distanceL1(double[] head, double[] label, double[] tail, int k){
        double sum = 0;
        for (int i = 0; i<k; i++){
            double calNumber = tail[i] - head[i] - label[i];
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

    public double distanceL2(double[] head, double[] label, double[] tail, int k){
        double sum = 0;
        for (int i = 0; i < k; i++){
            double calNumber = tail[i] - head[i] - label[i];
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

    public ArrayList<Triple<Integer, Integer, Integer>> sample(ArrayList<Triple<Integer, Integer, Integer>> triples, int size){
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

    /**
     * dot production
     * @param vector1
     * @param vector2
     * @return dot product of two vectors
     */

    public Double dotProduct(double[] vector1, double[] vector2){
        double sum =0;
        for (int i=0; i<vector1.length; i++){
            double temp = vector1[i] * vector2[i];
            sum = sum + temp;
        }
        return sum;
    }

    /**
     * check if the two vectors are orthogonal
     * @param vector
     * @param normalVector
     * @return boolean value
     */

    public Boolean checkOrthogonal(double[] vector, double[] normalVector){
        double dotProduct = dotProduct(vector, normalVector);
        double numerator = Math.abs(dotProduct);
        if (numerator <= epsilon){
            return true;
        }else {
            return false;
        }
    }

    /**
     * project a vector to a plane
     * @param vector
     * @param normalVector
     * @return project vector
     */
    public double[] planeProjection(double[] vector, double[] normalVector){
        double dotProduct = dotProduct(vector, normalVector);
        double[] projectedVector = new double[vector.length];
        for(int i = 0; i < vector.length; i++){
            double projectedNumber = vector[i]- dotProduct*normalVector[i];
            projectedVector[i] = projectedNumber;
        }
        return projectedVector;
    }

    /**
     * project a vector to a space
     * @param vector
     * @param matrix
     * @return project vector
     */

    public double[] spaceProjection(double[] vector, double[][] matrix){
        double[] projectedVector = new double[vector.length];
        for (int i =0; i < vector.length; i ++){
            double number = vector[i];
            double sum = 0;
            for (int j =0; j<matrix[0].length; j++){
                sum = sum + number * matrix[i][j];
            }
            projectedVector[i] = sum;
        }
        return projectedVector;
    }

    /**
     * initialize identity matrix with k rows and d columns
     * @param k
     * @param d
     * @return
     */
    public double[][] identityMatrix(int k, int d){
        double[][] matrix = new double[k][d];
        for (int i = 0; i < k; i ++){
            for (int j = 0; j< d; j++){
                if (i ==j){
                    matrix[i][j] = 1d;
                } else {
                    matrix[i][j] = 0d;
                }
            }
        }
        return matrix;
    }

    /**
     * norm of matrix
     * @param matrix
     * @return normed matrix
     */

    public double[][] normMatrix(double[][] matrix){
        double[][] normedMatrix = new double[matrix.length][matrix[0].length];
        SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.createRealMatrix(matrix));
        double norm = svd.getNorm();
        for (int i = 0; i < matrix.length; i ++){
            for (int j = 0; i < matrix[0].length; j++){
                normedMatrix[i][j] = matrix[i][j] / norm;
            }
        }
        return normedMatrix;
    }

    /**
     * euclidean distance between two vectors
     * @param vector1
     * @param vector2
     * @return euclidean distance
     */
    public double euclideanDistance (double[] vector1, double[] vector2){
        double sum = 0d;
        for (int i = 0; i < vector1.length; i++){
            sum = sum + Math.pow(vector1[i]-vector2[i], 2);
        }
        double distance = Math.sqrt(sum);
        return distance;
    }

    /**
     * calculate the center point of several points
     * @param points
     * @return center point vector
     */
    public double[] centerPoint(ArrayList<double[]> points){
        double[] center = new double[points.get(0).length];
        for (int i = 0; i < points.get(0).length; i ++){
            double sum = 0d;
            for (int j =0; j<points.size(); j++){
                sum = sum + points.get(j)[i];
            }
            double average = sum / points.size();
            center[i] = average;
        }
        return center;
    }

    /**
     * sum of vector element
     * @param vector
     * @return the sum
     */

    public double vectorSum(double[] vector){
        double sum = 0d;
        for (double number: vector){
            sum = sum + number;
        }
        return sum;
    }

    /**
     * distance between cluster=specific relation vector and original relation vector
     * @param relationC
     * @param relation
     * @return distance
     */
    public double relationDistanceL2(double[] relationC, double[] relation){
        double sum = 0d;
        for (int i = 0; i< relation.length; i++){
            sum = sum + Math.pow((relationC[i] - relation[i]), 2);
        }
        return sum;
    }

}
