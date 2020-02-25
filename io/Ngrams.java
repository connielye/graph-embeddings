package com.prime.common.io;

import java.util.ArrayList;
import java.util.logging.Logger;

/**
 * generate ngrams from entity String
 *
 */

public class Ngrams {

    final static Logger logger = Logger.getLogger(Ngrams.class.getName());
    ArrayList<String> entityList;
    int n;
    ArrayList<ArrayList<String>> unigramList;
    ArrayList<ArrayList<String>> ngramList;

    public Ngrams(ArrayList<String> entityList, int n){
        this.entityList = entityList;
        this.n = n;
        this.unigramList = generateUnigram();
        generateNgram();

    }

    private ArrayList<ArrayList<String>> generateUnigram(){
        ArrayList<ArrayList<String>> result = new ArrayList<>();
        for (int i =0; i < entityList.size(); i ++){
            String entityString = entityList.get(i).toLowerCase();
            String[] tokens = entityString.split("");
            ArrayList<String> unigram = new ArrayList<>();
            for (String tok: tokens){
                if(!tok.equals(" ")){
                    unigram.add(tok);
                }
            }
            result.add(unigram);
        }
        return result;
    }

    private ArrayList<ArrayList<String>> generateBigram(){
        ArrayList<ArrayList<String>> result = new ArrayList<>();
        for (int i =0; i<unigramList.size(); i ++){
            ArrayList<String> bigram = new ArrayList<>();
            ArrayList<String> unigram = unigramList.get(i);
            for (int j=0; j< unigram.size()-1; j++){
                String tok = unigram.get(j) + unigram.get(j+1);
                bigram.add(tok);
            }
            result.add(bigram);
        }
        return result;
    }

    private ArrayList<ArrayList<String>> generateTrigram(){
        ArrayList<ArrayList<String>> result = new ArrayList<>();
        for (int i = 0; i<unigramList.size()-2; i++){
            ArrayList<String> trigram = new ArrayList<>();
            ArrayList<String> unigram = unigramList.get(i);
            for (int j =0; j < unigram.size(); j++){
                String tok = unigram.get(j) + unigram.get(j+1) + unigram.get(j+2);
                trigram.add(tok);
            }
            result.add(trigram);
        }
        return result;
    }

    private void generateNgram(){
        if (n > 3){
            logger.warning("n cannot be greater than 3!");
            System.exit(0);
        }
        ngramList = unigramList;
        ArrayList<ArrayList<String>> bigramList = generateBigram();
        ArrayList<ArrayList<String>> trigramList = generateTrigram();
        if (n == 2){
            for (int i = 0; i < ngramList.size(); i++){
                ArrayList<String> ngram = ngramList.get(i);
                ArrayList<String> bigram = bigramList.get(i);
                ngram.addAll(bigram);
            }
        }
        if (n == 3){
            for (int j =0; j<ngramList.size(); j++){
                ArrayList<String> ngram = ngramList.get(j);
                ArrayList<String> bigram = bigramList.get(j);
                ArrayList<String> trigram = trigramList.get(j);
                ngram.addAll(bigram);
                ngram.addAll(trigram);
            }
        }
    }

    public ArrayList<ArrayList<String>> getNgramList() {
        return ngramList;
    }

    public static void main(String[] args){
        ArrayList<String> entity = new ArrayList<>();
        entity.add("Sony Music");
        entity.add("Sony Game");
        Ngrams ngrams = new Ngrams(entity, 2);
        ArrayList<ArrayList<String>> ngramList = ngrams.getNgramList();
        for (ArrayList<String> ngram: ngramList){
            System.out.println(ngram);
        }
    }
}
