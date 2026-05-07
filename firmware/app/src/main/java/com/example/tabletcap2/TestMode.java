package com.example.tabletcap2;

/**
 * Operating mode for the DropleX tablet app.
 *
 * Only RAW mode is included in this public release.
 * RAW mode streams the live capacitance heatmap from the maXTouch sensor
 * with no automatic liquid detection or classification.
 */
public enum TestMode {
    /**
     * Streams the real-time raw capacitance heatmap.
     * Every electrode value is displayed as a colour-mapped cell;
     * no blob detection or ML classification is performed.
     */
    RAW;

    public String getDescription() {
        return "Real-time raw capacitance heatmap";
    }
}
