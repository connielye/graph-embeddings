package com.prime.common.io;

import org.apache.commons.lang3.tuple.Triple;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * read in RDF triples and create three lists: entity list, relation list and triple list
 *
 */
public class ReadText {

    ArrayList<String> stringList;
    ArrayList<String> entityList;
    ArrayList<String> relationList;
    ArrayList<Triple<String, String, String>> tripleList;

    public ReadText(String tripleFile) throws IOException{

        this.stringList = readString(tripleFile);
        this.tripleList = new ArrayList<Triple<String, String, String>>();
        this.entityList = new ArrayList<String>();
        this.relationList = new ArrayList<String>();
        parseString();

    }

    private ArrayList<String> readString (String filename) throws IOException{
        ArrayList<String> result = new ArrayList<String>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "utf8"));
        String line;
        while ((line = reader.readLine()) != null){
            result.add(line);
        }

        return result;
    }

    /**
     * parse the triples to get entity and relation list.
     */
    private void parseString(){
        for (String line: stringList){
            String[] list = line.split("\t");
            String subject = list[0];
            String predicate = list[1];
            String object = list[2];
            tripleList.add(Triple.of(subject, predicate, object));
            if (! entityList.contains(subject)){
                entityList.add(subject);
            }
            if (! entityList.contains(object)){
                entityList.add(object);
            }
            if(! relationList.contains(predicate)){
                relationList.add(predicate);
            }
        }
    }

    public ArrayList<String> getEntityList() {
        return entityList;
    }

    public ArrayList<String> getRelationList() {
        return relationList;
    }

    public ArrayList<Triple<String, String, String>> getTripleList() {
        return tripleList;
    }
}