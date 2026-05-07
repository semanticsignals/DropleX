package com.example.tabletcap2;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.Log;
import android.view.View;

public class HeatmapView extends View {
    private static final String TAG = "HeatmapView";

    private float[][] data;
    private Paint paint;
    private Paint textPaint;
    private Paint gridPaint;

    private float minValue = 0;
    private float maxValue = 1;
    private boolean autoScale = true;
    private boolean showGrid = false;
    private boolean showValues = false; // Set to true to show numerical values
    private boolean showHeatmapColors = false; // Only show colors after classification
    private boolean debugMode = false; // Debug mode: always show colors and values for all cells
    private boolean showClassificationText = false; // Flag to enable/disable classification text labels

    // Grayscale color mapping (black -> white)
    private int[] colorMap;

    // Z-score thresholding and region selection
    private boolean[][] selectedRegions;
    private Paint selectionPaint;

    // Classification results for detected regions
    private java.util.List<RegionClassification> regionClassifications;
    private Paint classificationTextPaint;

    // Fake drop demo mode state
    private boolean fakeDropMode = false;
    private String pendingFakeDropLabel = null;
    private java.util.List<FakeDropAnnotation> fakeDropAnnotations;
    private FakeDropListener fakeDropListener;
    private static final int CALIBRATION_NOISE_SIZE = 6;   // shared 6x6 calibration corner size

    // Gaussian noise overlay for calibration corner (shared by FAKE_DROP, FAKE_CONT, DEMO_CAL)
    private boolean calNoiseEnabled = false;
    private float[][] calibrationNoise = null;
    private final java.util.Random calibRandom = new java.util.Random();

    // Fake container demo mode state
    private boolean fakeContMode = false;
    private boolean fakeContBoxesVisible = false;
    // Positions in data-space (52 rows × 32 cols after transpose).
    // In landscape view: left = high row index, right = low row index.
    // Moved inward from the edges (was 42 / 10) to bring boxes closer to screen centre.
    private static final int FAKE_CONT_LEFT_CENTER_ROW  = 36; // alcohol (left  in landscape)
    private static final int FAKE_CONT_LEFT_CENTER_COL  = 16; // centre vertically on screen
    private static final int FAKE_CONT_RIGHT_CENTER_ROW = 16; // water   (right in landscape)
    private static final int FAKE_CONT_RIGHT_CENTER_COL = 16; // centre vertically on screen
    private static final int FAKE_CONT_HALF = 6;              // 13x13 → half = 6

    // Fake container 3 demo mode state (single UNSAFE box, centre of screen)
    private boolean fakeContMode3 = false;
    private boolean fakeContBox3Visible = false;
    // Centre of the 52-row × 32-col data space (row=26, col=16)
    private static final int FAKE_CONT3_CENTER_ROW = 26;
    private static final int FAKE_CONT3_CENTER_COL = 16;
    // Reuses FAKE_CONT_HALF (=6) → 13×13 box, identical size to FAKE_CONT

    // Fake container 2 demo mode state (3 containers: water, water, DNA)
    private boolean fakeContMode2 = false;
    private int fakeCont2RevealCount = 0;  // 0=none, 1=box1 visible, 2=boxes 1&2, 3=all
    // 8×8 boxes evenly spaced across 52 rows.
    // In landscape: high row index = left side, low row index = right side.
    private static final int FAKE_CONT2_BOX_SIZE = 8; // 8×8 boxes
    private static final int FAKE_CONT2_COL      = 16; // vertically centred on 32-col axis
    private static final int FAKE_CONT2_BOX1_ROW = 39; // water 1 – leftmost  in landscape
    private static final int FAKE_CONT2_BOX2_ROW = 26; // water 2 – middle
    private static final int FAKE_CONT2_BOX3_ROW = 13; // DNA     – rightmost in landscape

    // Cache for patch bounds to avoid recomputing isWithinClassifiedPatch for every cell
    private boolean[][] patchMask;  // Pre-computed mask of which cells are in classified patches

    // Track thresholds for classification override
    private float positiveThreshold = 0;  // Positive z-score threshold (mean + z*std)
    private float negativeThreshold = 0;  // Negative z-score threshold (mean - z*std)

    // Custom red cells perimeter for demo mode
    private boolean showRedPerimeter = false;
    private boolean showRedCells2 = false;  // Toggle for customRedCells2
    
    // Container mode configuration
    private boolean containerMode = false;
    private int topContainerRow = 10;
    private int topContainerCol = 16;

    private int bottomContainerRow = 42;
    private int bottomContainerCol = 16;
    private static final int CONTAINER_REGION_SIZE = 13;
    private Paint containerRegionPaint;
    
    private static final int[][] CUSTOM_RED_CELLS = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7},
        {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7},
        {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7},
        {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7},
        {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6}, {4, 7},
        {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6}, {5, 7},
        {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7},
        {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}
    };
    private static final int[][] CUSTOM_RED_CELLS2 = {
//        {4,15},{10,15},{16,15},{22,15},{28,15},{34,15},{40,15},{46,15},
//        {4,21},{10,21},{16,21},{22,21},{28,21},{34,21},{40,21},{46,21},
//        {4,27},{10,27},{16,27},{22,27},{28,27},{34,27},{40,27},{46,27}
        {15,4},{15,10},{15,16},{15,22},{15,28},{15,34},{15,40},{15,46},
        {21,4},{21,10},{21,16},{21,22},{21,28},{21,34},{21,40},{21,46},
        {27,4},{27,10},{27,16},{27,22},{27,28},{27,34},{27,40},{27,46}
    };

    // Helper class to store classification results with position
    private static class RegionClassification {
        int centroidRow;
        int centroidCol;
        String prediction;
        float[] confidences;

        RegionClassification(int row, int col, String pred, float[] conf) {
            centroidRow = row;
            centroidCol = col;
            prediction = pred;
            confidences = conf;
        }
    }

    // Helper class for persistent fake-drop annotation boxes.
    private static class FakeDropAnnotation {
        int centroidRow;
        int centroidCol;
        String label;

        FakeDropAnnotation(int row, int col, String label) {
            this.centroidRow = row;
            this.centroidCol = col;
            this.label = label;
        }
    }

    public interface FakeDropListener {
        void onFakeDropCaptured(String label);
    }

    public HeatmapView(Context context) {
        super(context);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setStyle(Paint.Style.FILL);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(20);
        textPaint.setTextAlign(Paint.Align.CENTER);

        gridPaint = new Paint();
        gridPaint.setColor(Color.GRAY);
        gridPaint.setStrokeWidth(0.5f);
        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setAlpha(64);

        // Initialize selection paint for marking regions
        selectionPaint = new Paint();
        selectionPaint.setColor(Color.RED);
        selectionPaint.setStrokeWidth(3f);
        selectionPaint.setStyle(Paint.Style.STROKE);

        // Initialize classification text paint
        classificationTextPaint = new Paint();
        classificationTextPaint.setColor(Color.CYAN);
        classificationTextPaint.setTextSize(24);
        classificationTextPaint.setTextAlign(Paint.Align.CENTER);
        classificationTextPaint.setAntiAlias(true);
        classificationTextPaint.setShadowLayer(3.0f, 0, 0, Color.BLACK); // Add shadow for visibility

        // Generate grayscale color map
        generateColorMap();
    }

    private void generateColorMap() {
        colorMap = new int[256];

        // Simple diverging color map centered at 0:
        // Negative values (minValue to 0): Green
        // Positive values (0 to maxValue): Red
        // The center of the 256-index range (index 128) maps to value 0

        int centerIndex = 128;

        for (int i = 0; i < 256; i++) {
            if (i < centerIndex) {
                // Green for negative values (indices 0-127).
                // i=127 → white (ratio=1), i=0 → darkest green (ratio=0).
                // Apply gamma < 1 so mid-range negatives spread further from white,
                // making e.g. -400 visibly darker than -200.
                float ratio = (float) i / centerIndex;          // 0 (dark) → 1 (white)
                float curved = (float) Math.pow(ratio, 0.45f);  // gamma compress toward white
                int intensity = (int)(curved * 255);
                colorMap[i] = Color.rgb(intensity, 255, intensity);
            } else {
                // Red for positive values (indices 128-255)
                // i=128 is light red, i=255 is dark red
                float ratio = (float) (i - centerIndex) / (255 - centerIndex);
                int intensity = (int)((1 - ratio) * 255);
                colorMap[i] = Color.rgb(255, intensity, intensity);
            }
        }
    }

    public void setData(float[][] newData) {
        this.data = newData;

        // Regenerate calibration noise every frame whenever noise corner is active.
        if (calNoiseEnabled && newData != null && newData.length > 0) {
            regenerateCalibrationNoise(newData.length, newData[0].length);
        }

        // Don't reset display state - let autoSelectBlobs update it
        // This prevents flickering between blob detections
        // showHeatmapColors, selectedRegions, and regionClassifications
        // will be updated by autoSelectBlobs() if needed

        // Adjusted colormap range for better sensitivity
        // Typical values range from -200 to 250, so using -300 to 300 for full color spectrum
        minValue = -300.0f;
        maxValue = 300.0f;

        // Request redraw
        invalidate();
    }

    public void setColorRange(float min, float max) {
        this.minValue = min;
        this.maxValue = max;
        this.autoScale = false;
        invalidate();
    }

    public void setShowGrid(boolean show) {
        this.showGrid = show;
        invalidate();
    }

    public void setShowValues(boolean show) {
        this.showValues = show;
        invalidate();
    }

    public void setDebugMode(boolean enabled) {
        this.debugMode = enabled;
        invalidate(); // Redraw to show/hide debug info
        Log.e(TAG, "Debug mode " + (enabled ? "ENABLED" : "DISABLED"));
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public void setShowClassificationText(boolean enabled) {
        this.showClassificationText = enabled;
        invalidate(); // Redraw to show/hide classification text
        Log.e(TAG, "Classification text " + (enabled ? "ENABLED" : "DISABLED"));
    }

    public boolean isShowClassificationText() {
        return showClassificationText;
    }

    public void setShowRedPerimeter(boolean enabled) {
        this.showRedPerimeter = enabled;
        invalidate(); // Redraw to show/hide red perimeter
        Log.e(TAG, "Red perimeter " + (enabled ? "ENABLED" : "DISABLED"));
    }

    public void toggleRedCells2() {
        this.showRedCells2 = !this.showRedCells2;
        invalidate(); // Redraw to show/hide red cells 2
        Log.e(TAG, "Red cells 2 " + (showRedCells2 ? "ENABLED" : "DISABLED"));
    }

    public boolean isShowRedCells2() {
        return showRedCells2;
    }

    public void setContainerMode(boolean enabled, int topRow, int topCol, int bottomRow, int bottomCol) {
        this.containerMode = enabled;
        this.topContainerRow = topRow;
        this.topContainerCol = topCol;
        this.bottomContainerRow = bottomRow;
        this.bottomContainerCol = bottomCol;
        
        // Initialize paint for container regions if not already done
        if (containerRegionPaint == null) {
            containerRegionPaint = new Paint();
            containerRegionPaint.setColor(Color.YELLOW);
            containerRegionPaint.setStyle(Paint.Style.STROKE);
            containerRegionPaint.setStrokeWidth(4f);
        }
        
        invalidate();
        Log.e(TAG, "Container mode " + (enabled ? "ENABLED" : "DISABLED") + 
              " at positions Top(" + topRow + "," + topCol + ") Bottom(" + bottomRow + "," + bottomCol + ")");
    }

    public void setFakeDropMode(boolean enabled) {
        fakeDropMode = enabled;
        calNoiseEnabled = enabled;
        if (!enabled) {
            pendingFakeDropLabel = null;
            fakeDropListener = null;
            if (fakeDropAnnotations != null) {
                fakeDropAnnotations.clear();
            }
        } else if (fakeDropAnnotations == null) {
            fakeDropAnnotations = new java.util.ArrayList<>();
        }
        // FAKE_DROP should never show legacy red overlays.
        if (enabled) {
            showRedPerimeter = false;
            showRedCells2 = false;
        }
        invalidate();
    }

    public void setCalNoiseEnabled(boolean enabled) {
        calNoiseEnabled = enabled;
        if (!enabled) {
            calibrationNoise = null;
        }
        invalidate();
    }

    public void setFakeDropListener(FakeDropListener listener) {
        this.fakeDropListener = listener;
    }

    public void setFakeContMode(boolean enabled) {
        fakeContMode = enabled;
        calNoiseEnabled = enabled;
        fakeContBoxesVisible = false;
        if (enabled) {
            showRedPerimeter = false;
            showRedCells2 = false;
        }
        invalidate();
    }

    public void revealFakeContBoxes() {
        fakeContBoxesVisible = true;
        invalidate();
    }

    public void setFakeCont2Mode(boolean enabled) {
        fakeContMode2 = enabled;
        calNoiseEnabled = enabled;
        fakeCont2RevealCount = 0;
        if (enabled) {
            showRedPerimeter = false;
            showRedCells2 = false;
        }
        invalidate();
    }

    public void setFakeCont3Mode(boolean enabled) {
        fakeContMode3 = enabled;
        calNoiseEnabled = enabled;
        fakeContBox3Visible = false;
        if (enabled) {
            showRedPerimeter = false;
            showRedCells2 = false;
        }
        invalidate();
    }

    public void revealFakeCont3Box() {
        fakeContBox3Visible = true;
        invalidate();
    }

    /** Reveal the next container box (call up to 3 times). */
    public void revealFakeCont2Box() {
        if (fakeCont2RevealCount < 3) {
            fakeCont2RevealCount++;
        }
        invalidate();
    }

    public void armFakeDropLabel(String label) {
        if (label == null || label.trim().isEmpty()) {
            return;
        }
        pendingFakeDropLabel = label.trim();
        Log.e(TAG, "Fake drop armed for label: " + pendingFakeDropLabel);
    }

    public void resetFakeDropDemo() {
        pendingFakeDropLabel = null;
        if (fakeDropAnnotations != null) {
            fakeDropAnnotations.clear();
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (data == null || data.length == 0) {
            // Draw placeholder text
            canvas.drawColor(Color.BLACK);
            textPaint.setTextSize(40);
            canvas.drawText("No data to display", getWidth() / 2f, getHeight() / 2f, textPaint);
            return;
        }

        int rows = data.length;
        int cols = data[0].length;

        float cellWidth = (float) getWidth() / cols;
        float cellHeight = (float) getHeight() / rows;

        // Draw heatmap cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float value = data[i][j];
                boolean isCalibrationBlank = isInCalibrationBlankRegion(i, j, rows, cols);

                // Replace calibration cells with Gaussian noise so they look like real
                // (but tiny) sensor readings rather than blank white.
                if (isCalibrationBlank && calNoiseEnabled
                        && calibrationNoise != null
                        && i < calibrationNoise.length
                        && j < calibrationNoise[0].length) {
                    value = calibrationNoise[i][j];
                    isCalibrationBlank = false;  // render via normal colour path
                }

                boolean isNoise = false;
                
                // Filter out noise values > 1000, display as white
                if (value > 800) {
                    isNoise = true;
                }

                if (isCalibrationBlank) {
                    // Reserve a white area for calibration drop in FAKE_DROP mode.
                    paint.setColor(Color.WHITE);
                } else if (isNoise) {
                    // Display noise values > 1000 as white
                    paint.setColor(Color.WHITE);
                } else if (debugMode || containerMode) {
                    // Debug mode or Container mode: show colors for all cells
                    if (Float.isNaN(value) || Float.isInfinite(value)) {
                        paint.setColor(Color.GRAY);
                    } else {
                        // Normalize value to 0-255 range for diverging color map
                        // minValue (-1000) -> 0 (green)
                        // -250 -> 96 (white)
                        // maxValue (1000) -> 255 (red)
                        float normalized = (value - minValue) / (maxValue - minValue);
                        normalized = Math.max(0, Math.min(1, normalized)); // Clamp to [0, 1]

                        int colorIndex = (int)(normalized * 255);
                        paint.setColor(colorMap[colorIndex]);
                    }
                } else {
                    // Normal mode: only show colors within classified patches
                    if (isNoise) {
                        // Display noise values > 1000 as white
                        paint.setColor(Color.WHITE);
                    } else if (!showHeatmapColors || !isWithinClassifiedPatch(i, j)) {
                        paint.setColor(Color.WHITE);
                    } else if (Float.isNaN(value) || Float.isInfinite(value)) {
                        // Skip NaN or Infinite values
                        paint.setColor(Color.GRAY);
                    } else {
                        // Normalize value to 0-255 range for diverging color map
                        // minValue (-1000) -> 0 (green)
                        // -250 -> 96 (white)
                        // maxValue (1000) -> 255 (red)
                        float normalized = (value - minValue) / (maxValue - minValue);
                        normalized = Math.max(0, Math.min(1, normalized)); // Clamp to [0, 1]

                        int colorIndex = (int)(normalized * 255);
                        paint.setColor(colorMap[colorIndex]);
                    }
                }

                // Calculate cell position
                float left = j * cellWidth;
                float top = i * cellHeight;
                float right = left + cellWidth;
                float bottom = top + cellHeight;

                // Draw cell
                canvas.drawRect(left, top, right, bottom, paint);

                // Draw value text if enabled
                // In debug mode: show values for all cells
                // In normal mode: only show values within classified patches
                boolean shouldShowValue = debugMode ?
                    (showValues && !Float.isNaN(value) && !Float.isInfinite(value)) :
                    (showValues && !Float.isNaN(value) && !Float.isInfinite(value) && isWithinClassifiedPatch(i, j));
                shouldShowValue = shouldShowValue && !isCalibrationBlank;

                if (shouldShowValue) {
                    // Calculate text size based on cell dimensions
                    float textSize = Math.min(cellWidth, cellHeight) * 0.3f;
                    textPaint.setTextSize(textSize);

                    // Choose text color based on background color for diverging color map
                    // Dark green (0-30): white text
                    // Light green to white to light red (30-220): black text
                    // Dark red (220-255): white text
                    float normalized = (value - minValue) / (maxValue - minValue);
                    int colorIndex = (int)(normalized * 255);
                    if (colorIndex < 30 || colorIndex > 220) {
                        textPaint.setColor(Color.WHITE);
                    } else {
                        textPaint.setColor(Color.BLACK);
                    }

                    // Format the number with appropriate precision
                    String text;
                    if (Math.abs(value) >= 100) {
                        text = String.format("%.0f", value);
                    } else if (Math.abs(value) >= 10) {
                        text = String.format("%.1f", value);
                    } else {
                        text = String.format("%.1f", value);
                    }

                    // Rotate text for landscape mode
                    canvas.save();
                    float centerX = left + cellWidth / 2;
                    float centerY = top + cellHeight / 2;
                    canvas.rotate(-90, centerX, centerY); // Rotate 90 degrees counter-clockwise
                    canvas.drawText(text,
                                  centerX,
                                  centerY + textPaint.getTextSize() / 3,
                                  textPaint);
                    canvas.restore();
                }
            }
        }

        // Draw grid lines if enabled
        if (showGrid) {
            for (int i = 0; i <= rows; i++) {
                float y = i * cellHeight;
                canvas.drawLine(0, y, getWidth(), y, gridPaint);
            }

            for (int j = 0; j <= cols; j++) {
                float x = j * cellWidth;
                canvas.drawLine(x, 0, x, getHeight(), gridPaint);
            }
        }

        // Draw selected regions overlay
        if (!fakeDropMode && selectedRegions != null && selectedRegions.length > 0) {
            drawSelectedRegions(canvas, cellWidth, cellHeight);
        }

        // Draw classification results for detected regions (only if showClassificationText is enabled)
        if (showClassificationText && regionClassifications != null && !regionClassifications.isEmpty()) {
            drawClassificationLabels(canvas, cellWidth, cellHeight);
        }

        // Draw red perimeter of custom cells in demo mode
        if (!fakeDropMode && showRedPerimeter && rows > 0 && cols > 0) {
            drawRedPerimeter(canvas, cellWidth, cellHeight, rows, cols);
        }

        // Draw red cells 2 markers in demo mode
        if (!fakeDropMode && showRedCells2 && rows > 0 && cols > 0) {
            drawRedCells2(canvas, cellWidth, cellHeight, rows, cols);
        }

        // Draw container regions in container mode
        if (containerMode && rows > 0 && cols > 0) {
            drawContainerRegions(canvas, cellWidth, cellHeight, rows, cols);
        }

        if (fakeDropMode && fakeDropAnnotations != null && !fakeDropAnnotations.isEmpty()) {
            drawFakeDropAnnotations(canvas, cellWidth, cellHeight, rows, cols);
        }

        if (fakeContMode && fakeContBoxesVisible) {
            drawFakeContBoxes(canvas, cellWidth, cellHeight, rows, cols);
        }

        if (fakeContMode2 && fakeCont2RevealCount > 0) {
            drawFakeCont2Boxes(canvas, cellWidth, cellHeight, rows, cols);
        }

        if (fakeContMode3 && fakeContBox3Visible) {
            drawFakeCont3Box(canvas, cellWidth, cellHeight, rows, cols);
        }

        // Draw color scale legend at the bottom
        // drawColorScale(canvas);  // Removed colorbar
    }

    /**
     * Draw container region overlays
     */
    private void drawContainerRegions(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        if (containerRegionPaint == null) return;

        int halfSize = CONTAINER_REGION_SIZE / 2;

        // Draw top container region
        if (topContainerRow >= 0 && topContainerRow < rows && topContainerCol >= 0 && topContainerCol < cols) {
            float left = (topContainerCol - halfSize) * cellWidth;
            float top = (topContainerRow - halfSize) * cellHeight;
            float right = (topContainerCol + halfSize + 1) * cellWidth;
            float bottom = (topContainerRow + halfSize + 1) * cellHeight;
            canvas.drawRect(left, top, right, bottom, containerRegionPaint);
        }

        // Draw bottom container region
        if (bottomContainerRow >= 0 && bottomContainerRow < rows && bottomContainerCol >= 0 && bottomContainerCol < cols) {
            float left = (bottomContainerCol - halfSize) * cellWidth;
            float top = (bottomContainerRow - halfSize) * cellHeight;
            float right = (bottomContainerCol + halfSize + 1) * cellWidth;
            float bottom = (bottomContainerRow + halfSize + 1) * cellHeight;
            canvas.drawRect(left, top, right, bottom, containerRegionPaint);
        }
    }

    private boolean isInCalibrationBlankRegion(int row, int col, int totalRows, int totalCols) {
        if (!calNoiseEnabled) {
            return false;
        }
        // "Top-right" in landscape = bottom-right in portrait draw coordinates.
        int startRow = Math.max(0, totalRows - CALIBRATION_NOISE_SIZE);
        int startCol = Math.max(0, totalCols - CALIBRATION_NOISE_SIZE);
        return row >= startRow && col >= startCol;
    }

    private void regenerateCalibrationNoise(int rows, int cols) {
        if (calibrationNoise == null
                || calibrationNoise.length != rows
                || calibrationNoise[0].length != cols) {
            calibrationNoise = new float[rows][cols];
        }
        int startRow = Math.max(0, rows - CALIBRATION_NOISE_SIZE);
        int startCol = Math.max(0, cols - CALIBRATION_NOISE_SIZE);
        for (int i = startRow; i < rows; i++) {
            for (int j = startCol; j < cols; j++) {
                // Gaussian σ≈7 → ~99 % of values within ±20
                float v = (float) (calibRandom.nextGaussian() * 7.0);
                calibrationNoise[i][j] = Math.max(-20f, Math.min(20f, v));
            }
        }
    }

    /**
     * Draw red perimeter around custom red cells region.
     * Note: The heatmap is transposed, so we need to swap row/col when drawing.
     */
    private void drawRedPerimeter(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        // Create a set for fast lookup of customRedCells
        java.util.Set<String> cellSet = new java.util.HashSet<>();
        for (int[] cell : CUSTOM_RED_CELLS) {
            // Flip coordinates: swap row and col
            int transposedRow = cell[1];
            int transposedCol = cell[0];
            cellSet.add(transposedRow + "," + transposedCol);
        }

        // Draw red borders only for perimeter cells
        Paint perimeterPaint = new Paint();
        perimeterPaint.setColor(Color.RED);
        perimeterPaint.setStrokeWidth(4f);
        perimeterPaint.setStyle(Paint.Style.STROKE);

        for (int[] cell : CUSTOM_RED_CELLS) {
            // Flip coordinates
            int transposedRow = cell[1];
            int transposedCol = cell[0];

            // Validate cell coordinates
            if (transposedRow >= 0 && transposedRow < rows && transposedCol >= 0 && transposedCol < cols) {
                float left = transposedCol * cellWidth;
                float top = transposedRow * cellHeight;
                float right = left + cellWidth;
                float bottom = top + cellHeight;

                // Check if this cell is on the perimeter
                boolean isPerimeter = false;

                // Check 4 neighbors (up, down, left, right)
                if (!cellSet.contains((transposedRow - 1) + "," + transposedCol) ||  // top neighbor
                    !cellSet.contains((transposedRow + 1) + "," + transposedCol) ||  // bottom neighbor
                    !cellSet.contains(transposedRow + "," + (transposedCol - 1)) ||  // left neighbor
                    !cellSet.contains(transposedRow + "," + (transposedCol + 1))) {  // right neighbor
                    isPerimeter = true;
                }

                // Draw border if this is a perimeter cell
                if (isPerimeter) {
                    canvas.drawRect(left, top, right, bottom, perimeterPaint);
                }
            }
        }
    }

    /**
     * Draw red markers for customRedCells2.
     * These are individual cells, not a perimeter.
     */
    private void drawRedCells2(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        // Paint for red cell markers
        Paint redCellPaint = new Paint();
        redCellPaint.setColor(Color.RED);
        redCellPaint.setStyle(Paint.Style.FILL);

        for (int[] cell : CUSTOM_RED_CELLS2) {
            // Flip coordinates: swap row and col
            int transposedRow = cell[1];
            int transposedCol = cell[0];

            // Validate cell coordinates
            if (transposedRow >= 0 && transposedRow < rows && transposedCol >= 0 && transposedCol < cols) {
                float left = transposedCol * cellWidth;
                float top = transposedRow * cellHeight;
                float right = left + cellWidth;
                float bottom = top + cellHeight;

                // Fill the cell with red
                canvas.drawRect(left, top, right, bottom, redCellPaint);
            }
        }
    }

    private void drawFakeContBoxes(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        Paint borderPaint = new Paint();
        borderPaint.setColor(Color.BLACK);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(6f);

        Paint labelTextPaint = new Paint();
        labelTextPaint.setColor(Color.BLACK);
        labelTextPaint.setAntiAlias(true);
        labelTextPaint.setTextAlign(Paint.Align.CENTER);
        labelTextPaint.setFakeBoldText(true);
        labelTextPaint.setTextSize(Math.min(cellWidth, cellHeight) * 1.8f);

        drawContainerBox(canvas, cellWidth, cellHeight, rows, cols,
                FAKE_CONT_LEFT_CENTER_ROW, FAKE_CONT_LEFT_CENTER_COL, FAKE_CONT_HALF,
                "DON'T DRINK", borderPaint, labelTextPaint);
        drawContainerBox(canvas, cellWidth, cellHeight, rows, cols,
                FAKE_CONT_RIGHT_CENTER_ROW, FAKE_CONT_RIGHT_CENTER_COL, FAKE_CONT_HALF,
                "SAFE", borderPaint, labelTextPaint);
    }

    private void drawFakeCont2Boxes(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        Paint borderPaint = new Paint();
        borderPaint.setColor(Color.BLACK);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(6f);

        Paint labelTextPaint = new Paint();
        labelTextPaint.setColor(Color.BLACK);
        labelTextPaint.setAntiAlias(true);
        labelTextPaint.setTextAlign(Paint.Align.CENTER);
        labelTextPaint.setFakeBoldText(true);
        labelTextPaint.setTextSize(Math.min(cellWidth, cellHeight) * 1.8f);

        if (fakeCont2RevealCount >= 1) {
            drawFakeCont2SingleBox(canvas, cellWidth, cellHeight, rows, cols,
                    FAKE_CONT2_BOX1_ROW, FAKE_CONT2_COL, "WATER", borderPaint, labelTextPaint);
        }
        if (fakeCont2RevealCount >= 2) {
            drawFakeCont2SingleBox(canvas, cellWidth, cellHeight, rows, cols,
                    FAKE_CONT2_BOX2_ROW, FAKE_CONT2_COL, "WATER", borderPaint, labelTextPaint);
        }
        if (fakeCont2RevealCount >= 3) {
            drawFakeCont2SingleBox(canvas, cellWidth, cellHeight, rows, cols,
                    FAKE_CONT2_BOX3_ROW, FAKE_CONT2_COL, "DNA", borderPaint, labelTextPaint);
        }
    }

    private void drawFakeCont3Box(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        Paint borderPaint = new Paint();
        borderPaint.setColor(Color.BLACK);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(6f);

        Paint labelTextPaint = new Paint();
        labelTextPaint.setColor(Color.BLACK);
        labelTextPaint.setAntiAlias(true);
        labelTextPaint.setTextAlign(Paint.Align.CENTER);
        labelTextPaint.setFakeBoldText(true);
        labelTextPaint.setTextSize(Math.min(cellWidth, cellHeight) * 1.8f);

        drawContainerBox(canvas, cellWidth, cellHeight, rows, cols,
                FAKE_CONT3_CENTER_ROW, FAKE_CONT3_CENTER_COL, FAKE_CONT_HALF,
                "UNSAFE", borderPaint, labelTextPaint);
    }

    /** 8×8 box renderer for FAKE_CONT2. Uses an even-sized box centred on the given cell. */
    private void drawFakeCont2SingleBox(Canvas canvas, float cellWidth, float cellHeight,
                                        int rows, int cols,
                                        int centerRow, int centerCol, String label,
                                        Paint borderPaint, Paint labelTextPaint) {
        int half = FAKE_CONT2_BOX_SIZE / 2;  // = 4
        int startRow = Math.max(0, centerRow - half);
        int endRow   = Math.min(rows, startRow + FAKE_CONT2_BOX_SIZE);
        int startCol = Math.max(0, centerCol - half);
        int endCol   = Math.min(cols, startCol + FAKE_CONT2_BOX_SIZE);

        float boxLeft   = startCol * cellWidth;
        float boxTop    = startRow * cellHeight;
        float boxRight  = endCol   * cellWidth;
        float boxBottom = endRow   * cellHeight;

        canvas.drawRect(boxLeft, boxTop, boxRight, boxBottom, borderPaint);

        if (label == null || label.isEmpty()) return;

        float ascent  = labelTextPaint.ascent();
        float descent = labelTextPaint.descent();
        float sideClearance = Math.max(16f, cellWidth * 0.9f);
        float textX = boxRight + sideClearance - ascent;
        float textY = (boxTop + boxBottom) / 2f;

        textY += Math.max(10f, cellHeight * 0.7f);
        textY = Math.max(labelTextPaint.getTextSize(), Math.min(getHeight() - 8f, textY));

        if (textX > getWidth() - 8f) {
            textX = boxLeft - sideClearance - descent;
        }

        canvas.save();
        canvas.rotate(-90, textX, textY);
        canvas.drawText(label, textX, textY, labelTextPaint);
        canvas.restore();
    }

    /** Shared box renderer used by FAKE_CONT. */
    private void drawContainerBox(Canvas canvas, float cellWidth, float cellHeight,
                                  int rows, int cols,
                                  int centerRow, int centerCol, int half, String label,
                                  Paint borderPaint, Paint labelTextPaint) {
        int startRow = Math.max(0, centerRow - half);
        int endRow   = Math.min(rows, centerRow + half + 1);
        int startCol = Math.max(0, centerCol - half);
        int endCol   = Math.min(cols, centerCol + half + 1);

        float boxLeft   = startCol * cellWidth;
        float boxTop    = startRow * cellHeight;
        float boxRight  = endCol   * cellWidth;
        float boxBottom = endRow   * cellHeight;

        canvas.drawRect(boxLeft, boxTop, boxRight, boxBottom, borderPaint);

        if (label == null || label.isEmpty()) return;

        // Place label BELOW the box in landscape view.
        // "Below in landscape" = to the RIGHT in portrait coordinates (higher x).
        float ascent  = labelTextPaint.ascent();   // negative
        float descent = labelTextPaint.descent();  // positive
        float sideClearance = Math.max(16f, cellWidth * 0.9f);
        float textX = boxRight + sideClearance - ascent;
        float textY = (boxTop + boxBottom) / 2f;

        // Shift slightly downward in landscape (positive portrait-Y direction).
        textY += Math.max(10f, cellHeight * 0.7f);
        textY = Math.max(labelTextPaint.getTextSize(), Math.min(getHeight() - 8f, textY));

        if (textX > getWidth() - 8f) {
            textX = boxLeft - sideClearance - descent;
        }

        canvas.save();
        canvas.rotate(-90, textX, textY);
        canvas.drawText(label, textX, textY, labelTextPaint);
        canvas.restore();
    }

    private void drawFakeDropAnnotations(Canvas canvas, float cellWidth, float cellHeight, int rows, int cols) {
        Paint borderPaint = new Paint();
        borderPaint.setColor(Color.BLACK);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(6f);

        Paint labelTextPaint = new Paint();
        labelTextPaint.setColor(Color.BLACK);
        labelTextPaint.setAntiAlias(true);
        labelTextPaint.setTextAlign(Paint.Align.CENTER);
        labelTextPaint.setFakeBoldText(true);
        labelTextPaint.setTextSize(Math.min(cellWidth, cellHeight) * 1.5f);

        final int patchSize = 6;
        final int offsetBeforeRow = 2;
        final int offsetBeforeCol = 3;
        Rect textBounds = new Rect();

        for (FakeDropAnnotation annotation : fakeDropAnnotations) {
            int startRow = annotation.centroidRow - offsetBeforeRow;
            int startCol = annotation.centroidCol - offsetBeforeCol;
            int endRow = startRow + patchSize;
            int endCol = startCol + patchSize;

            startRow = Math.max(0, startRow);
            startCol = Math.max(0, startCol);
            endRow = Math.min(rows, endRow);
            endCol = Math.min(cols, endCol);

            float boxLeft = startCol * cellWidth;
            float boxTop = startRow * cellHeight;
            float boxRight = endCol * cellWidth;
            float boxBottom = endRow * cellHeight;

            canvas.drawRect(boxLeft, boxTop, boxRight, boxBottom, borderPaint);

            String label = annotation.label == null ? "" : annotation.label.toUpperCase();
            if (label.isEmpty()) {
                continue;
            }

            labelTextPaint.getTextBounds(label, 0, label.length(), textBounds);

            // The map/value text is rendered with a -90 rotation for landscape readability.
            // To place label "below" from that same landscape viewpoint, we place anchor
            // on the RIGHT in portrait coordinates, then rotate text by -90.
            // Use ascent/descent-aware offset so glyphs never touch the black box border.
            float sideClearance = Math.max(14f, cellWidth * 0.7f);
            float ascent = labelTextPaint.ascent();   // negative
            float descent = labelTextPaint.descent(); // positive
            float textX = boxRight + sideClearance - ascent;
            float textY = (boxTop + boxBottom) / 2f;

            // Slightly lower in landscape view (vertical shift in portrait coordinates).
            float lowerOffset = Math.max(15f, cellHeight * 0.8f);
            textY += lowerOffset;

            // Keep text on-screen: if too far right, place to left side instead,
            // while preserving clearance from the box edge.
            if (textX > getWidth() - 8f) {
                textX = boxLeft - sideClearance - descent;
            }

            // Keep text vertically within screen bounds.
            float minY = Math.max(textBounds.height(), 8f);
            float maxY = getHeight() - 8f;
            textY = Math.max(minY, Math.min(maxY, textY));

            // Plain text only (no white background / no label border).
            canvas.save();
            canvas.rotate(-90, textX, textY);
            canvas.drawText(label, textX, textY, labelTextPaint);
            canvas.restore();
        }
    }

    /**
     * Check if a cell (row, col) is within any classified patch.
     * Uses pre-computed patchMask for O(1) lookup instead of O(n) computation.
     */
    private boolean isWithinClassifiedPatch(int row, int col) {
        if (patchMask == null || row >= patchMask.length || col >= patchMask[0].length) {
            return false;
        }
        return patchMask[row][col];
    }

    /**
     * Pre-compute which cells are within classified patches.
     * This is called once after classification to build a lookup table.
     */
    private void buildPatchMask() {
        if (data == null || data.length == 0) {
            patchMask = null;
            return;
        }

        int rows = data.length;
        int cols = data[0].length;
        patchMask = new boolean[rows][cols];

        if (regionClassifications == null || regionClassifications.isEmpty()) {
            return;
        }

        final int PATCH_SIZE = 6;
        java.util.List<java.util.List<int[]>> regions = separateRegions(selectedRegions);

        for (RegionClassification rc : regionClassifications) {
            for (java.util.List<int[]> region : regions) {
                // Check if this region contains the centroid
                boolean matchesCentroid = false;
                for (int[] pixel : region) {
                    if (pixel[0] == rc.centroidRow && pixel[1] == rc.centroidCol) {
                        matchesCentroid = true;
                        break;
                    }
                }

                if (!matchesCentroid) continue;

                // Calculate region dimensions
                int minRow = Integer.MAX_VALUE, maxRow = Integer.MIN_VALUE;
                int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
                for (int[] pixel : region) {
                    minRow = Math.min(minRow, pixel[0]);
                    maxRow = Math.max(maxRow, pixel[0]);
                    minCol = Math.min(minCol, pixel[1]);
                    maxCol = Math.max(maxCol, pixel[1]);
                }
                int regionHeight = maxRow - minRow + 1;
                int regionWidth = maxCol - minCol + 1;

                // Calculate offset (same as classifyDetectedRegions)
                int offsetBeforeRow, offsetBeforeCol;
                if (regionHeight != regionWidth) {
                    offsetBeforeRow = (PATCH_SIZE > 2) ? (PATCH_SIZE / 2 - 1) : 0;
                    offsetBeforeCol = (PATCH_SIZE > 2) ? (PATCH_SIZE / 2 + 1) : 0;
                } else {
                    offsetBeforeRow = (PATCH_SIZE / 2) - 1;
                    offsetBeforeCol = PATCH_SIZE / 2;
                }

                // Calculate patch bounds
                int startRow = rc.centroidRow - offsetBeforeRow;
                int startCol = rc.centroidCol - offsetBeforeCol;
                int endRow = startRow + PATCH_SIZE;
                int endCol = startCol + PATCH_SIZE;

                // Mark all cells in this patch
                for (int r = startRow; r < endRow && r < rows; r++) {
                    if (r < 0) continue;
                    for (int c = startCol; c < endCol && c < cols; c++) {
                        if (c >= 0) {
                            patchMask[r][c] = true;
                        }
                    }
                }
            }
        }
    }

    /**
     * Draw classification result labels for each detected region.
     * Each label is displayed above the patch (in landscape orientation).
     */
    private void drawClassificationLabels(Canvas canvas, float cellWidth, float cellHeight) {
        if (regionClassifications == null || data == null) {
            return;
        }

        int rows = data.length;
        int cols = data[0].length;

        // Draw label for each classified region
        for (RegionClassification rc : regionClassifications) {
            // Calculate position for landscape view with -90 degree rotation:
            // To appear "above" in landscape, position to the left in portrait coordinates
            // For a 6x6 patch centered at centroid with offsetBefore=3:
            // - Patch left edge is at (centroidCol - 3) in terms of cells
            // - Place label 1.5 cells to the left of the patch left edge
            float x = (rc.centroidCol - 3f) * cellWidth; // 1.5 cells to the left of patch
            float y = rc.centroidRow * cellHeight + cellHeight / 2; // vertically centered on centroid

            // Draw the label with readable text size (larger and in all caps)
            float textSize = Math.min(cellWidth, cellHeight) * 1.2f;
            classificationTextPaint.setTextSize(textSize);
            classificationTextPaint.setTextAlign(Paint.Align.CENTER);

            // Rotate text for landscape mode
            canvas.save();
            canvas.rotate(-90, x, y); // Rotate 90 degrees counter-clockwise around the label position
            canvas.drawText(rc.prediction.toUpperCase(), x, y, classificationTextPaint);
            canvas.restore();
        }
    }

    /**
     * Draw overlay showing selected regions and 6x6 patch boundaries.
     */
    private void drawSelectedRegions(Canvas canvas, float cellWidth, float cellHeight) {
        if (selectedRegions == null || data == null) {
            return;
        }

        int rows = selectedRegions.length;
        int cols = selectedRegions[0].length;

        // Draw red borders around each selected pixel
        selectionPaint.setColor(Color.RED);
        selectionPaint.setStrokeWidth(2f);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (selectedRegions[i][j]) {
                    // Draw rectangle border around this pixel
                    float left = j * cellWidth;
                    float top = i * cellHeight;
                    float right = left + cellWidth;
                    float bottom = top + cellHeight;

                    canvas.drawRect(left, top, right, bottom, selectionPaint);
                }
            }
        }

        // Draw 6x6 patch boundaries around each separate region
        java.util.List<java.util.List<int[]>> separateRegions = separateRegions(selectedRegions);

        // Paint for 6x6 patch boundaries (lime green dashed)
        Paint patchPaint = new Paint();
        patchPaint.setColor(Color.rgb(0, 0, 0)); // Black
        patchPaint.setStrokeWidth(4f);
        patchPaint.setStyle(Paint.Style.STROKE);

        // Create dashed line effect
        patchPaint.setPathEffect(new android.graphics.DashPathEffect(new float[]{10, 10}, 0));

        for (java.util.List<int[]> region : separateRegions) {
            // Calculate centroid of this region
            float sumRow = 0;
            float sumCol = 0;
            for (int[] pixel : region) {
                sumRow += pixel[0];
                sumCol += pixel[1];
            }
            int centroidRow = Math.round(sumRow / region.size());
            int centroidCol = Math.round(sumCol / region.size());

            // Calculate region dimensions for adaptive centering
            int minRow = Integer.MAX_VALUE, maxRow = Integer.MIN_VALUE;
            int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
            for (int[] pixel : region) {
                minRow = Math.min(minRow, pixel[0]);
                maxRow = Math.max(maxRow, pixel[0]);
                minCol = Math.min(minCol, pixel[1]);
                maxCol = Math.max(maxCol, pixel[1]);
            }
            int regionHeight = maxRow - minRow + 1;
            int regionWidth = maxCol - minCol + 1;
            int pixelCount = region.size();

            // Calculate 6x6 patch bounds with adaptive centering (matching classifyDetectedRegions)
            int patchSize = 6;
            int offsetBeforeRow;
            int offsetBeforeCol;

            // Choose offset strategy (same as classifyDetectedRegions):
            // - For rectangular regions (height != width): shift right by 1 (offsetBeforeRow=2, offsetBeforeCol=4)
            // - For square regions (height == width): use offsetBeforeRow=2, offsetBeforeCol=3
            if (regionHeight != regionWidth) {
                // Rectangular region: shift right by 1
                offsetBeforeRow = (patchSize > 2) ? (patchSize / 2 - 1) : 0;  // For size=6, this is 2
                offsetBeforeCol = (patchSize > 2) ? (patchSize / 2) : 0;  // For size=6, this is 4
            } else {
                // Square region: center symmetrically
                offsetBeforeRow = patchSize / 2;  // For size=6, this is 2
                offsetBeforeCol = patchSize / 2;        // For size=6, this is 3
            }

            // Convert to pixel coordinates
            float left = (centroidCol - offsetBeforeCol) * cellWidth;
            float top = (centroidRow - offsetBeforeRow) * cellHeight;
            float right = left + patchSize * cellWidth;
            float bottom = top + patchSize * cellHeight;

            // Draw 6x6 patch boundary
            canvas.drawRect(left, top, right, bottom, patchPaint);
        }
    }

    /**
     * Separate a binary mask into individual connected components.
     * Returns a list of regions, where each region is a list of pixel coordinates [row, col].
     */
    private java.util.List<java.util.List<int[]>> separateRegions(boolean[][] mask) {
        int rows = mask.length;
        int cols = mask[0].length;

        boolean[][] visited = new boolean[rows][cols];
        java.util.List<java.util.List<int[]>> regions = new java.util.ArrayList<>();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mask[i][j] && !visited[i][j]) {
                    java.util.List<int[]> region = new java.util.ArrayList<>();
                    floodFill(mask, visited, i, j, region);
                    regions.add(region);
                }
            }
        }

        return regions;
    }

    private void drawColorScale(Canvas canvas) {
        int legendHeight = 40;
        int legendY = getHeight() - legendHeight - 20;
        int legendWidth = getWidth() - 100;
        int legendX = 50;

        // Draw color gradient
        for (int i = 0; i < legendWidth; i++) {
            float ratio = (float) i / legendWidth;
            int colorIndex = (int)(ratio * 255);
            paint.setColor(colorMap[colorIndex]);
            canvas.drawLine(legendX + i, legendY, legendX + i, legendY + legendHeight, paint);
        }

        // Draw border
        gridPaint.setStrokeWidth(2);
        canvas.drawRect(legendX, legendY, legendX + legendWidth, legendY + legendHeight, gridPaint);

        // Draw min/max labels
        textPaint.setTextSize(25);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextAlign(Paint.Align.LEFT);
        canvas.drawText(String.format("%.2f", minValue), legendX, legendY + legendHeight + 25, textPaint);

        textPaint.setTextAlign(Paint.Align.RIGHT);
        canvas.drawText(String.format("%.2f", maxValue), legendX + legendWidth, legendY + legendHeight + 25, textPaint);

        textPaint.setTextAlign(Paint.Align.CENTER);
        canvas.drawText(String.format("%.2f", (minValue + maxValue) / 2),
                       legendX + legendWidth / 2, legendY + legendHeight + 25, textPaint);
    }

    /**
     * Calculate z-score threshold value for the data.
     * Similar to measure.py's zscore_threshold function.
     *
     * @param zThreshold Number of standard deviations from mean
     * @param mode "positive" - above mean, "negative" - below mean, "both" - either direction
     * @return The computed threshold value
     */
    private float calculateZScoreThreshold(float zThreshold, String mode) {
        if (data == null || data.length == 0) {
            return 0;
        }

        // Calculate mean and standard deviation
        float sum = 0;
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                float val = data[i][j];
                if (!Float.isNaN(val) && !Float.isInfinite(val)) {
                    sum += val;
                    count++;
                }
            }
        }

        float mean = sum / count;

        // Calculate standard deviation
        float sumSquaredDiff = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                float val = data[i][j];
                if (!Float.isNaN(val) && !Float.isInfinite(val)) {
                    float diff = val - mean;
                    sumSquaredDiff += diff * diff;
                }
            }
        }
        float std = (float) Math.sqrt(sumSquaredDiff / count);

        float threshold;
        if ("positive".equals(mode)) {
            threshold = mean + zThreshold * std;
        } else if ("negative".equals(mode)) {
            threshold = mean - zThreshold * std;
        } else { // "both" - use whichever direction has stronger signal
            // Check which direction has more extreme values
            float maxPos = Float.MIN_VALUE;
            float maxNeg = Float.MAX_VALUE;

            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    float val = data[i][j];
                    if (!Float.isNaN(val) && !Float.isInfinite(val)) {
                        maxPos = Math.max(maxPos, val - mean);
                        maxNeg = Math.min(maxNeg, val - mean);
                    }
                }
            }

            if (maxPos > Math.abs(maxNeg)) {
                threshold = mean + zThreshold * std;
            } else {
                threshold = mean - zThreshold * std;
            }
        }

        return threshold;
    }

    /**
     * Automatically select prominent blobs using Z-score thresholding.
     * Similar to measure.py's auto_select_blobs function.
     *
     * @param zThreshold Z-score threshold value (default 2.0)
     * @param minSize Minimum number of pixels for a blob to be kept
     */
    public void autoSelectBlobs(float zThreshold, int minSize) {
        if (data == null || data.length == 0) {
            return;
        }

        int rows = data.length;
        int cols = data[0].length;

        // Calculate mean and standard deviation for logging
        float sum = 0;
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float val = data[i][j];
                if (!Float.isNaN(val) && !Float.isInfinite(val)) {
                    sum += val;
                    count++;
                }
            }
        }
        float mean = sum / count;

        // Calculate standard deviation
        float sumSquaredDiff = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float val = data[i][j];
                if (!Float.isNaN(val) && !Float.isInfinite(val)) {
                    float diff = val - mean;
                    sumSquaredDiff += diff * diff;
                }
            }
        }
        float std = (float) Math.sqrt(sumSquaredDiff / count);

        // Calculate both thresholds for logging and classification
        this.positiveThreshold = mean + (zThreshold * std);
        this.negativeThreshold = mean - (zThreshold * std);

        // Log threshold information
        Log.e(TAG, "[THRESHOLD] Z-score=" + zThreshold + ", Mean=" + String.format("%.2f", mean) +
                   ", Std=" + String.format("%.2f", std) +
                   ", Positive=" + String.format("%.2f", positiveThreshold) +
                   ", Negative=" + String.format("%.2f", negativeThreshold) +
                   ", Using=BOTH (detecting in both directions)");

        // Create binary mask detecting blobs in BOTH directions
        boolean[][] mask = new boolean[rows][cols];
        java.util.List<Float> selectedValues = new java.util.ArrayList<>();

        // Select pixels that are either:
        // 1. Significantly above mean (> positive threshold), OR
        // 2. Significantly below mean (< negative threshold)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Calibration corner is excluded from FAKE_DROP detection candidates.
                if (isInCalibrationBlankRegion(i, j, rows, cols)) {
                    continue;
                }
                float val = data[i][j];
                if (!Float.isNaN(val) && !Float.isInfinite(val)) {
                    if (val > positiveThreshold || val < negativeThreshold) {
                        mask[i][j] = true;
                        selectedValues.add(val);
                    }
                }
            }
        }

        // Log selected pixel values
        if (!selectedValues.isEmpty()) {
            float minSelected = Float.MAX_VALUE;
            float maxSelected = Float.MIN_VALUE;
            float sumSelected = 0;
            for (float val : selectedValues) {
                minSelected = Math.min(minSelected, val);
                maxSelected = Math.max(maxSelected, val);
                sumSelected += val;
            }
            float meanSelected = sumSelected / selectedValues.size();

            Log.e(TAG, "[SELECTED] " + selectedValues.size() + " pixels selected - " +
                       "Min=" + String.format("%.2f", minSelected) +
                       ", Max=" + String.format("%.2f", maxSelected) +
                       ", Mean=" + String.format("%.2f", meanSelected));
        } else {
            Log.e(TAG, "[SELECTED] 0 pixels selected");
        }

        // Remove small blobs using connected components (8-connectivity)
        mask = removeSmallBlobs(mask, minSize);

        // Remove blobs with bad dimensions (height and width differ by more than 2)
        mask = removeBadDimensionBlobs(mask, 2);

        selectedRegions = mask;

        // Separate into individual regions to count blobs
        java.util.List<java.util.List<int[]>> regions = separateRegions(selectedRegions);
        int numBlobs = regions.size();

        Log.e(TAG, "[BLOBS] Detected " + numBlobs + " blob(s) after filtering");

        // Classify each detected region
        boolean hasClassifications = classifyDetectedRegions();

        // Enable heatmap colors only if at least one region was classified
        showHeatmapColors = hasClassifications;

        // If no blobs detected, clear the regions and classifications
        if (numBlobs == 0) {
            selectedRegions = null;
            regionClassifications = null;
            showHeatmapColors = false;
            patchMask = null;
        }

        // Build the patch mask for fast lookup during drawing
        buildPatchMask();

        // Trigger redraw to show selected regions and classifications
        invalidate();

        if (fakeDropMode && pendingFakeDropLabel != null && regionClassifications != null && !regionClassifications.isEmpty()) {
            String armedLabel = pendingFakeDropLabel;
            boolean requireNonOverlap = "salt water".equalsIgnoreCase(armedLabel);
            RegionClassification selected = null;
            for (RegionClassification rc : regionClassifications) {
                if (!isOverlappingExistingFakeDropPatch(rc.centroidRow, rc.centroidCol)) {
                    selected = rc;
                    break;
                }
            }

            // For salt-water capture, never allow overlap with existing water region.
            // Keep waiting until a non-overlapping candidate appears.
            if (selected == null && requireNonOverlap) {
                return;
            }
            if (selected == null) {
                selected = regionClassifications.get(0);
            }

            if (fakeDropAnnotations == null) {
                fakeDropAnnotations = new java.util.ArrayList<>();
            }

            String capturedLabel = pendingFakeDropLabel;
            fakeDropAnnotations.add(new FakeDropAnnotation(
                selected.centroidRow,
                selected.centroidCol,
                capturedLabel
            ));
            pendingFakeDropLabel = null;
            invalidate();

            if (fakeDropListener != null) {
                fakeDropListener.onFakeDropCaptured(capturedLabel);
            }
        }
    }

    private boolean isOverlappingExistingFakeDropPatch(int centroidRow, int centroidCol) {
        if (fakeDropAnnotations == null || fakeDropAnnotations.isEmpty()) {
            return false;
        }

        final int patchSize = 6;
        final int offsetBeforeRow = 2;
        final int offsetBeforeCol = 3;

        int startRow = centroidRow - offsetBeforeRow;
        int endRow = startRow + patchSize - 1;
        int startCol = centroidCol - offsetBeforeCol;
        int endCol = startCol + patchSize - 1;

        for (FakeDropAnnotation annotation : fakeDropAnnotations) {
            int otherStartRow = annotation.centroidRow - offsetBeforeRow;
            int otherEndRow = otherStartRow + patchSize - 1;
            int otherStartCol = annotation.centroidCol - offsetBeforeCol;
            int otherEndCol = otherStartCol + patchSize - 1;

            boolean rowsOverlap = startRow <= otherEndRow && endRow >= otherStartRow;
            boolean colsOverlap = startCol <= otherEndCol && endCol >= otherStartCol;
            if (rowsOverlap && colsOverlap) {
                return true;
            }
        }
        return false;
    }

    private boolean isNearExistingFakeDrop(int row, int col, int minDistanceCells) {
        if (fakeDropAnnotations == null || fakeDropAnnotations.isEmpty()) {
            return false;
        }
        for (FakeDropAnnotation annotation : fakeDropAnnotations) {
            int dRow = annotation.centroidRow - row;
            int dCol = annotation.centroidCol - col;
            double distance = Math.sqrt(dRow * dRow + dCol * dCol);
            if (distance <= minDistanceCells) {
                return true;
            }
        }
        return false;
    }

    /**
     * Clear all auto-detection overlays and classification artifacts.
     */
    public void clearAutoDetection() {
        selectedRegions = null;
        regionClassifications = null;
        showHeatmapColors = false;
        patchMask = null;
        invalidate();
    }

    /**
     * Classify detected regions. Not used in RAW mode — no classifier is attached.
     * Kept as a stub so that autoSelectBlobs() compiles unchanged.
     */
    private boolean classifyDetectedRegions() {
        return false;
    }

    /**
     * Remove small connected components from a binary mask.
     * Uses 8-connectivity (includes diagonals).
     *
     * @param mask Input binary mask
     * @param minSize Minimum size (in pixels) to keep
     * @return Filtered mask with small components removed
     */
    private boolean[][] removeSmallBlobs(boolean[][] mask, int minSize) {
        int rows = mask.length;
        int cols = mask[0].length;

        boolean[][] result = new boolean[rows][cols];
        boolean[][] visited = new boolean[rows][cols];

        // For each unvisited true pixel, perform flood fill to find connected component
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mask[i][j] && !visited[i][j]) {
                    // Find connected component
                    java.util.List<int[]> component = new java.util.ArrayList<>();
                    floodFill(mask, visited, i, j, component);

                    // If component is large enough, add it to result
                    if (component.size() >= minSize) {
                        for (int[] pixel : component) {
                            result[pixel[0]][pixel[1]] = true;
                        }
                    }
                }
            }
        }

        return result;
    }

    /**
     * Remove blobs where height and width dimensions differ by more than maxDiff.
     * Uses 8-connectivity (includes diagonals).
     *
     * @param mask Input binary mask
     * @param maxDiff Maximum allowed difference between height and width
     * @return Filtered mask with bad dimension blobs removed
     */
    private boolean[][] removeBadDimensionBlobs(boolean[][] mask, int maxDiff) {
        int rows = mask.length;
        int cols = mask[0].length;

        boolean[][] result = new boolean[rows][cols];
        boolean[][] visited = new boolean[rows][cols];

        // For each unvisited true pixel, perform flood fill to find connected component
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mask[i][j] && !visited[i][j]) {
                    // Find connected component
                    java.util.List<int[]> component = new java.util.ArrayList<>();
                    floodFill(mask, visited, i, j, component);

                    // Calculate bounding box dimensions
                    int minRow = Integer.MAX_VALUE, maxRow = Integer.MIN_VALUE;
                    int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
                    for (int[] pixel : component) {
                        minRow = Math.min(minRow, pixel[0]);
                        maxRow = Math.max(maxRow, pixel[0]);
                        minCol = Math.min(minCol, pixel[1]);
                        maxCol = Math.max(maxCol, pixel[1]);
                    }
                    int height = maxRow - minRow + 1;
                    int width = maxCol - minCol + 1;
                    int dimensionDiff = Math.abs(height - width);

                    // If dimension difference is acceptable, add to result
                    if (dimensionDiff <= maxDiff) {
                        for (int[] pixel : component) {
                            result[pixel[0]][pixel[1]] = true;
                        }
                    } else {
                        Log.e(TAG, "[FILTER] Blob REJECTED: dimension difference too large (height=" + height + ", width=" + width + ", diff=" + dimensionDiff + ")");
                    }
                }
            }
        }

        return result;
    }

    /**
     * Flood fill algorithm to find connected components.
     * Uses 8-connectivity (includes diagonals).
     */
    private void floodFill(boolean[][] mask, boolean[][] visited, int startRow, int startCol,
                          java.util.List<int[]> component) {
        int rows = mask.length;
        int cols = mask[0].length;

        java.util.Queue<int[]> queue = new java.util.LinkedList<>();
        queue.add(new int[]{startRow, startCol});
        visited[startRow][startCol] = true;

        // 8-connected neighbors (including diagonals)
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1},
                             {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

        while (!queue.isEmpty()) {
            int[] pixel = queue.poll();
            component.add(pixel);

            int row = pixel[0];
            int col = pixel[1];

            // Check all 8 neighbors
            for (int[] dir : directions) {
                int newRow = row + dir[0];
                int newCol = col + dir[1];

                // Check bounds
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
                    if (mask[newRow][newCol] && !visited[newRow][newCol]) {
                        visited[newRow][newCol] = true;
                        queue.add(new int[]{newRow, newCol});
                    }
                }
            }
        }
    }
}