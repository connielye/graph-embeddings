package com.prime.common.io;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;

public class SerializeModelLists implements Serializable {
    Map<String, ArrayList<double[]>> map;

    public SerializeModelLists( Map<String, ArrayList<double[]>> map){
        this.map = map;
    }
}
