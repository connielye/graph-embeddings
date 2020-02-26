package com.prime.common.io;


import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import java.io.*;

/**
 * write serialized model to local disk
 */

public class WriteModel {

    private final static Logger logger = LogManager.getLogger(WriteModel.class);


    public void write(String outputFile, SerializeModelVectors model){
        try {
            FileOutputStream file = new FileOutputStream(outputFile);
            ObjectOutputStream outputStream = new ObjectOutputStream(file);
            outputStream.writeObject(model);
            outputStream.flush();

            outputStream.close();
            file.close();

        } catch (IOException e) {
            logger.error(e.getMessage());
        }
    }

    public void write2 (String outputFile, SerializeModelLists model){
        try {
            FileOutputStream file = new FileOutputStream(outputFile);
            ObjectOutputStream outputStream = new ObjectOutputStream(file);
            outputStream.writeObject(model);
            outputStream.flush();

            outputStream.close();
            file.close();

        } catch (IOException e) {
            logger.error(e.getMessage());
        }
    }
}



