package com.prime.common.io;

import org.apache.commons.lang3.tuple.Triple;
import org.apache.log4j.Logger;

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

    private final static Logger logger = Logger.getLogger(ReadText.class);
    ArrayList<String> stringList;
    ArrayList<String> entityList;
    ArrayList<String> relationList;
    ArrayList<Triple<String, String, String>> tripleList;

    public ReadText(String tripleFile, String entities, String relations) throws IOException{

        this.stringList = readString(tripleFile);
        this.entityList = readString(entities);
        this.relationList = readString(relations);
        this.tripleList = parseString();

    }

    private ArrayList<String> readString (String filename){
        ArrayList<String> result = new ArrayList<String>();

        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "utf8"));
            String line;
            while ((line = reader.readLine()) != null){
                result.add(line);
            }
        } catch (IOException e) {
            logger.error(e.getMessage());
        }

        return result;
    }

    /**
     * parse the triples to get entity and relation list.
     */
    private ArrayList<Triple<String, String, String>> parseString(){
        ArrayList<Triple<String, String, String>> triples = new ArrayList<Triple<String, String, String>>();
        for (String line: stringList){
            String[] list = line.split("\t");
            String subject = list[0];
            String predicate = list[1];
            String object = list[2];
            triples.add(Triple.of(subject, predicate, object));
        }
        return triples;
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