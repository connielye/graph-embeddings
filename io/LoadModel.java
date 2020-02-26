package com.prime.common.io;

import org.apache.log4j.Logger;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class LoadModel {

    private final static Logger logger = Logger.getLogger(LoadModel.class);

    public Map<String, double[]> load(String inputFile){
        Map<String, double[]> resultMap = new HashMap<String, double[]>();
        try {
            FileInputStream file = new FileInputStream(inputFile);
            ObjectInputStream inputStream = new ObjectInputStream(file);

            SerializeModelVectors model = (SerializeModelVectors)inputStream.readObject();

            resultMap = model.map;

            inputStream.close();
            file.close();
        } catch (IOException e) {
            logger.error(e.getMessage());
        } catch (ClassNotFoundException e) {
            logger.error(e.getMessage());
        }
        return resultMap;
    }

}
