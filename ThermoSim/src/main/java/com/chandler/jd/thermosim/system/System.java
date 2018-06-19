/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.chandler.jd.thermosim.system;

/**
 * Euclidian space for simulation. 0,0,0 is origin of coordinate system. Units
 * will be in meters.
 * 
 * @author darin
 */
public class System {
      
    private double xExtent;
      
    private double yExtent;
      
    private double zExtent;
    
    private double resolution;

    /**
     * @return the xExtent
     */
    public double getxExtent() {
        return xExtent;
    }

    /**
     * @param xExtent the xExtent to set
     */
    public void setxExtent(double xExtent) {
        this.xExtent = xExtent;
    }

    /**
     * @return the yExtent
     */
    public double getyExtent() {
        return yExtent;
    }

    /**
     * @param yExtent the yExtent to set
     */
    public void setyExtent(double yExtent) {
        this.yExtent = yExtent;
    }

    /**
     * @return the zExtent
     */
    public double getzExtent() {
        return zExtent;
    }

    /**
     * @param zExtent the zExtent to set
     */
    public void setzExtent(double zExtent) {
        this.zExtent = zExtent;
    }

    /**
     * @return the resolution. This is in m. The idea is that
     * the system will be evaluated in each axis direction in discrete steps, i.e.
     * finite element analysis. This should be the size of each discrete step.
     */
    public double getResolution() {
        return resolution;
    }

    /**
     * @param resolution the resolution to set. this is in m. The idea is that
     * the system will be evaluated in each axis direction in discrete steps, i.e.
     * finite element analysis. This should be the size of each discrete step.
     */
    public void setResolution(double resolution) {
        this.resolution = resolution;
    }

    
}
