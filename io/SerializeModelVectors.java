package com.prime.common.io;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;

/**
 * serialize model
 */

public class SerializeModelVectors implements Serializable{
    Map<String, double[]> map;

    public SerializeModelVectors(Map<String, double[]> map){
        this.map = map;
    }
}
