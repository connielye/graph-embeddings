package com.prime.common.io;

import org.apache.commons.lang3.tuple.Triple;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


/**
 * convert individuals string into index
 *
 */
public class Node2Id {

    Map<String, Integer> entity2id, relation2Id;
    ArrayList<Triple<Integer, Integer, Integer>> tripleIds;
    ArrayList<Integer> headList, lableList, tailList;
    int entitySize, relationSize;


    public Node2Id(ArrayList<String> entityList, ArrayList<String> relationList, ArrayList<Triple<String, String, String>> tripleList){

        this.entity2id = string2Id(entityList);
        this.relation2Id = string2Id(relationList);
        this.tripleIds = triple2Id(tripleList);
        this.headList = new ArrayList<Integer>();
        this.lableList = new ArrayList<Integer>();
        this.tailList = new ArrayList<Integer>();
        splitTripleIds();
        this.entitySize = entityList.size();
        this.relationSize = relationList.size();

    }

    /**
     * convert String into map key: String, value index number
     * @param inputList
     * @return
     */
    private Map<String, Integer> string2Id(ArrayList<String> inputList){

        Map<String, Integer> map = new HashMap<String, Integer>();
        for (int i = 0; i < inputList.size(); i ++){
            map.put(inputList.get(i), i);
        }

        return map;
    }

    /**
     * convert triple Strings to triple integers
     * @param inputList
     * @return
     */
    private ArrayList<Triple<Integer, Integer, Integer>> triple2Id (ArrayList<Triple<String, String, String>> inputList) {
        ArrayList<Triple<Integer, Integer, Integer>> triples = new ArrayList<Triple<Integer, Integer, Integer>>();
        for (Triple<String, String, String> stringTriple: inputList){
            String headString = stringTriple.getLeft();
            String labelString = stringTriple.getMiddle();
            String tailString = stringTriple.getRight();
            int head = entity2id.get(headString);
            int label = relation2Id.get(labelString);
            int tail = entity2id.get(tailString);
            triples.add(Triple.of(head, label, tail));
        }

        return triples;
    }

    private void splitTripleIds(){
        for(Triple<Integer, Integer, Integer> triple : tripleIds){
            headList.add(triple.getLeft());
            lableList.add(triple.getMiddle());
            tailList.add(triple.getRight());
        }
    }

    public ArrayList<Integer> getHeadList() {
        return headList;
    }

    public ArrayList<Integer> getLableList() {
        return lableList;
    }

    public ArrayList<Integer> getTailList() {
        return tailList;
    }

    public ArrayList<Triple<Integer, Integer, Integer>> getTripleIds() {
        return tripleIds;
    }

    public Map<String, Integer> getEntity2id() {
        return entity2id;
    }

    public Map<String, Integer> getRelation2Id() {
        return relation2Id;
    }

    public int getEntitySize() {
        return entitySize;
    }

    public int getRelationSize() {
        return relationSize;
    }
}

