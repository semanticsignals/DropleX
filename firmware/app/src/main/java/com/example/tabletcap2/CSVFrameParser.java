package com.example.tabletcap2;

import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVFrameParser {
    private static final String TAG = "CSVFrameParser";
    private static final int ROWS = 52;
    private static final int COLS = 32;

    // Configurable last frame number - MUST be set by calling setLastFrameNumber() before use
    // See MainActivity.LAST_FRAME_NUMBER for the configuration location
    private static int LAST_FRAME_NUMBER;

    /**
     * Set the last frame number to use for processing
     * This MUST be called before processing any frames
     * @param lastFrame The last frame number to use (1-based)
     */
    public static void setLastFrameNumber(int lastFrame) {
        LAST_FRAME_NUMBER = lastFrame;
        Log.e(TAG, "Last frame number set to: " + lastFrame);
    }

    /**
     * Process frames according to the formula:
     * frame_to_process = (delta_last + ref) - (delta_first + ref)
     * This simplifies to: frame_to_process = delta_last - delta_first
     *
     * @param deltaPath Path to the delta CSV file
     * @param refPath Path to the reference CSV file
     * @return 2D array [32][52] representing the processed frame
     */
    public static float[][] processFrameDifference(String deltaPath, String refPath) {
        try {
            // Step 1: Extract ref
            Log.e(TAG, "Extracting reference from: " + refPath);
            float[][] ref = parseReferenceFrame(refPath);
            if (ref == null) {
                Log.e(TAG, "Failed to parse reference frame");
                return null;
            }

            // Step 2: Extract delta_first (frame 1) and configured delta_last
            Log.e(TAG, "Extracting frames 1 and " + LAST_FRAME_NUMBER + " from: " + deltaPath);
            float[][][] deltaFrames = parseFirstAndLastFrames(deltaPath);
            if (deltaFrames == null) {
                Log.e(TAG, "Failed to parse delta frames");
                return null;
            }

            float[][] delta_first = deltaFrames[0];
            float[][] delta_last = deltaFrames[1];

            // Debug print raw delta frames
            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: Raw delta_first (frame 1) - first 5x5:");
            printMatrixDebug(delta_first, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(delta_first) +
                       ", max=" + getMax(delta_first) +
                       ", mean=" + getMean(delta_first));

            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: Raw delta_last (frame " + LAST_FRAME_NUMBER + ") - first 5x5:");
            printMatrixDebug(delta_last, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(delta_last) +
                       ", max=" + getMax(delta_last) +
                       ", mean=" + getMean(delta_last));

            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: Reference frame - first 5x5:");
            printMatrixDebug(ref, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(ref) +
                       ", max=" + getMax(ref) +
                       ", mean=" + getMean(ref));

            Log.e(TAG, "Processing: frame = (delta_last + ref) - (delta_first + ref)");

            // Step 3: delta_first += ref
            float[][] delta_first_plus_ref = new float[COLS][ROWS];
            for (int i = 0; i < COLS; i++) {
                for (int j = 0; j < ROWS; j++) {
                    delta_first_plus_ref[i][j] = delta_first[i][j] + ref[i][j];
                }
            }

            // Debug print delta_first_plus_ref
            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: delta_first + ref (first 5x5):");
            printMatrixDebug(delta_first_plus_ref, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(delta_first_plus_ref) +
                       ", max=" + getMax(delta_first_plus_ref) +
                       ", mean=" + getMean(delta_first_plus_ref));

            // Step 4: delta_last += ref
            float[][] delta_last_plus_ref = new float[COLS][ROWS];
            for (int i = 0; i < COLS; i++) {
                for (int j = 0; j < ROWS; j++) {
                    delta_last_plus_ref[i][j] = delta_last[i][j] + ref[i][j];
                }
            }

            // Debug print delta_last_plus_ref
            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: delta_last + ref (first 5x5):");
            printMatrixDebug(delta_last_plus_ref, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(delta_last_plus_ref) +
                       ", max=" + getMax(delta_last_plus_ref) +
                       ", mean=" + getMean(delta_last_plus_ref));

            // Step 5: frame_to_process = delta_last_plus_ref - delta_first_plus_ref
            // Note: This mathematically simplifies to delta_last - delta_first
            float[][] frame_to_process = new float[COLS][ROWS];
            for (int i = 0; i < COLS; i++) {
                for (int j = 0; j < ROWS; j++) {
                    frame_to_process[i][j] = delta_last_plus_ref[i][j] - delta_first_plus_ref[i][j];
                }
            }

            // Debug print frame_to_process before normalization
            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: frame_to_process BEFORE normalization (first 5x5):");
            printMatrixDebug(frame_to_process, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(frame_to_process) +
                       ", max=" + getMax(frame_to_process) +
                       ", mean=" + getMean(frame_to_process));

            // Step 6: Normalize - find maximum value (TEMPORARILY DISABLED)
            // float maxValue = 290;

            // Step 7: Subtract max value so all values are <= 0 (TEMPORARILY DISABLED)
            // Log.e(TAG, "========================================");
            // Log.e(TAG, "Normalizing: subtracting max value " + maxValue + " from all values");
            // for (int i = 0; i < COLS; i++) {
            //     for (int j = 0; j < ROWS; j++) {
            //         frame_to_process[i][j] = frame_to_process[i][j] - maxValue;
            //     }
            // }

            // Debug print frame_to_process after normalization
            Log.e(TAG, "========================================");
            Log.e(TAG, "DEBUG: frame_to_process AFTER normalization (first 5x5):");
            printMatrixDebug(frame_to_process, 5, 5);
            Log.e(TAG, "Stats: min=" + getMin(frame_to_process) +
                       ", max=" + getMax(frame_to_process) +
                       ", mean=" + getMean(frame_to_process));
            Log.e(TAG, "========================================");

            return frame_to_process;

        } catch (Exception e) {
            Log.e(TAG, "Error during frame difference processing: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Parse first and last frames from a delta CSV file with proper normalization
     * Following the exact normalization from measure.py:
     * 1. measured = ref + delta (for each frame)
     * 2. measured -= measured[0] (subtract first frame from all frames)
     * 3. measured -= max(measured) (normalize so all values are negative)
     *
     * @param deltaPath Path to the delta CSV file
     * @param refPath Path to the reference CSV file
     * @return Array of 2 frames [2][32][52] representing normalized first and last frames
     */
    public static float[][][] parseFirstAndLastFramesNormalized(String deltaPath, String refPath) {
        try {
            // Parse reference frame
//            Log.e(TAG, "Parsing reference from: " + refPath);
            float[][] refFrame = parseReferenceFrame(refPath);
            if (refFrame == null) {
//                Log.e(TAG, "Failed to parse reference frame");
                return null;
            }

            // Parse first and last delta frames
//            Log.e(TAG, "Parsing first and last frames from: " + deltaPath);
            float[][][] deltaFrames = parseFirstAndLastFrames(deltaPath);
            if (deltaFrames == null) {
//                Log.e(TAG, "Failed to parse delta frames");
                return null;
            }

            // Apply normalization procedure from measure.py
            // Step 1: measured = ref + delta for each frame
            float[][][] measured = new float[2][COLS][ROWS];
            for (int frame = 0; frame < 2; frame++) {
                for (int i = 0; i < COLS; i++) {
                    for (int j = 0; j < ROWS; j++) {
                        measured[frame][i][j] = refFrame[i][j] + deltaFrames[frame][i][j];
                    }
                }
            }

            // Step 2: measured -= measured[0] (subtract first frame from all frames)
            // This makes the first frame all zeros, and the last frame shows the difference
            for (int frame = 0; frame < 2; frame++) {
                for (int i = 0; i < COLS; i++) {
                    for (int j = 0; j < ROWS; j++) {
                        measured[frame][i][j] = measured[frame][i][j] - measured[0][i][j];
                    }
                }
            }

            // Step 3: Find maximum value across both frames
            float maxValue = Float.NEGATIVE_INFINITY;
            for (int frame = 0; frame < 2; frame++) {
                for (int i = 0; i < COLS; i++) {
                    for (int j = 0; j < ROWS; j++) {
                        if (measured[frame][i][j] > maxValue) {
                            maxValue = measured[frame][i][j];
                        }
                    }
                }
            }

            // Step 4: Normalize to maximum value so all values are negative
            for (int frame = 0; frame < 2; frame++) {
                for (int i = 0; i < COLS; i++) {
                    for (int j = 0; j < ROWS; j++) {
                        measured[frame][i][j] = measured[frame][i][j] - maxValue;
                    }
                }
            }

            Log.e(TAG, "Normalization complete. Max value was: " + maxValue);
            Log.e(TAG, "First frame should be all zeros or negative");
            Log.e(TAG, "Last frame range: [" + getMin(measured[1]) + ", " + getMax(measured[1]) + "]");

            return measured;

        } catch (Exception e) {
            Log.e(TAG, "Error during normalized parsing: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Parse ALL frames from a CSV file
     * @param filepath Path to the CSV file
     * @return 3D array [numFrames][32][52] with all frames
     */
    public static float[][][] parseAllFrames(String filepath) {
        try {
            File file = new File(filepath);
            if (!file.exists()) {
                Log.e(TAG, "File does not exist: " + filepath);
                return null;
            }

            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            List<float[][]> frames = new ArrayList<>();
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                // Skip the header row
                if (lineNumber == 1) {
//                    Log.e(TAG, "Skipping header row");
                    continue;
                }

                // Parse each data row as a frame
                float[][] frame = parseRow(line);
                if (frame != null) {
                    frames.add(frame);
                }
            }

            reader.close();

            if (frames.isEmpty()) {
                Log.e(TAG, "No frames found in CSV");
                return null;
            }

            Log.e(TAG, "Successfully parsed " + frames.size() + " frames");

            // Convert List to array
            float[][][] result = new float[frames.size()][][];
            for (int i = 0; i < frames.size(); i++) {
                result[i] = frames.get(i);
            }

            return result;

        } catch (IOException e) {
            Log.e(TAG, "Error reading CSV file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Parse first frame and configured last frame from a CSV file
     * @param filepath Path to the CSV file
     * @return 3D array [2][32][52] with first frame at index 0 and configured last frame at index 1
     */
    public static float[][][] parseFirstAndLastFrames(String filepath) {
        try {
            File file = new File(filepath);
            if (!file.exists()) {
                Log.e(TAG, "File does not exist: " + filepath);
                return null;
            }

            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            String specifiedLastLine = null;
            float[][] firstFrame = null;
            int lineNumber = 0;

            // Calculate which line number we need for the last frame
            int lastTargetLine = LAST_FRAME_NUMBER + 1; // +1 for header

            Log.e(TAG, "Looking for frames: 1 (first) and " + LAST_FRAME_NUMBER);

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                // Skip the header row
                if (lineNumber == 1) {
//                    Log.e(TAG, "Skipping header row");
                    continue;
                }

                // Process the first data frame (line 2)
                if (lineNumber == 2) {
//                    Log.e(TAG, "Processing first frame (line " + lineNumber + ")");
                    firstFrame = parseRow(line);
                }

                // Check if this is the specified last frame
                if (lineNumber == lastTargetLine) {
                    Log.e(TAG, "Processing specified frame " + LAST_FRAME_NUMBER + " (line " + lineNumber + ")");
                    specifiedLastLine = line;
                    // We can break here since we have both frames
                    break;
                }
            }

            reader.close();

            if (firstFrame == null || specifiedLastLine == null) {
                Log.e(TAG, "Failed to find requested frames. First frame: " +
                           (firstFrame != null ? "found" : "not found") +
                           ", Frame " + LAST_FRAME_NUMBER + ": " +
                           (specifiedLastLine != null ? "found" : "not found"));
                return null;
            }

            // Parse the specified last frame
            Log.e(TAG, "Processing frame " + LAST_FRAME_NUMBER + " as last frame");
            float[][] lastFrame = parseRow(specifiedLastLine);

            // Return both frames
            float[][][] frames = new float[2][][];
            frames[0] = firstFrame;
            frames[1] = lastFrame;

            Log.e(TAG, "Successfully parsed frames 1 and " + LAST_FRAME_NUMBER);
            return frames;

        } catch (IOException e) {
            Log.e(TAG, "Error reading CSV file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Helper to get maximum value from a 2D array
     */
    private static float getMax(float[][] matrix) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > max) {
                    max = matrix[i][j];
                }
            }
        }
        return max;
    }

    /**
     * Parse and normalize the first frame from a delta CSV file with reference subtraction
     * Following the exact normalization from measure.py:
     * 1. measured = ref + delta
     * 2. measured -= measured[0] (subtract first frame)
     * 3. measured -= max(measured) (normalize so all values are negative)
     *
     * @param deltaPath Path to the delta CSV file
     * @param refPath Path to the reference CSV file
     * @return 2D array [32][52] representing the normalized first frame, or null on error
     */
    public static float[][] parseFirstFrameNormalized(String deltaPath, String refPath) {
        try {
            // Parse reference frame
//            Log.e(TAG, "Parsing reference from: " + refPath);
            float[][] refFrame = parseReferenceFrame(refPath);
            if (refFrame == null) {
                Log.e(TAG, "Failed to parse reference frame");
                return null;
            }

            // Parse current delta frame
//            Log.e(TAG, "Parsing delta from: " + deltaPath);
            float[][] deltaFrame = parseFirstFrame(deltaPath);
            if (deltaFrame == null) {
                Log.e(TAG, "Failed to parse delta frame");
                return null;
            }

            // Parse delta_0 frame (initial baseline)
            // Look in the same directory as the current delta file
            java.io.File deltaFile = new java.io.File(deltaPath);
            java.io.File deltaDir = deltaFile.getParentFile();
            String delta0Path = new java.io.File(deltaDir, "deltas_0.csv").getAbsolutePath();
//            Log.e(TAG, "Parsing delta_0 from: " + delta0Path);
            float[][] delta0Frame = parseFirstFrame(delta0Path);
            if (delta0Frame == null) {
                Log.e(TAG, "Failed to parse delta_0 frame from: " + delta0Path);
                return null;
            }

            // Apply normalization procedure from measure.py
            // Step 1: Add ref to delta_0 to get baseline
            float[][] delta0Baseline = new float[COLS][ROWS];
            for (int i = 0; i < COLS; i++) {
                for (int j = 0; j < ROWS; j++) {
                    delta0Baseline[i][j] = delta0Frame[i][j] + refFrame[i][j];
                }
            }

            // Step 2: measured = ref + delta
            float[][] measured = new float[COLS][ROWS];
            for (int i = 0; i < COLS; i++) {
                for (int j = 0; j < ROWS; j++) {
                    measured[i][j] = refFrame[i][j] + deltaFrame[i][j];
                }
            }

            // Step 3: Subtract delta_0 baseline (measured -= delta_0_baseline)
            for (int i = 0; i < COLS; i++) {
                for (int j = 0; j < ROWS; j++) {
                    measured[i][j] = measured[i][j] - delta0Baseline[i][j];
                }
            }

            // Step 3: Normalize using hardcoded max value (TEMPORARILY DISABLED)
            // float maxValue = 290;

            // Step 4: Normalize to maximum value so all values are negative (TEMPORARILY DISABLED)
            // for (int i = 0; i < COLS; i++) {
            //     for (int j = 0; j < ROWS; j++) {
            //         measured[i][j] = measured[i][j] - maxValue;
            //     }
            // }

            return measured;

        } catch (Exception e) {
            Log.e(TAG, "Error during normalized parsing: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Parse the reference frame from ref.csv
     * @param filepath Path to the reference CSV file
     * @return 2D array [32][52] representing the reference frame after rotation
     */
    public static float[][] parseReferenceFrame(String filepath) {
        try {
            File file = new File(filepath);
            if (!file.exists()) {
                Log.e(TAG, "Reference file does not exist: " + filepath);
                return null;
            }

            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                // Skip the header row
                if (lineNumber == 1) {
//                    Log.e(TAG, "Skipping reference header row");
                    continue;
                }

                // Process the second row (reference data)
                if (lineNumber == 2) {
//                    Log.e(TAG, "Processing reference data");
                    float[][] frame = parseRow(line);
                    reader.close();
                    return frame;
                }
            }

            reader.close();
//            Log.e(TAG, "No data rows found in reference CSV");
            return null;

        } catch (IOException e) {
            Log.e(TAG, "Error reading reference CSV file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Parse the first frame from a CSV file following the maXTouch format
     * @param filepath Path to the CSV file
     * @return 2D array [32][52] representing the first frame after rotation, or null on error
     */
    public static float[][] parseFirstFrame(String filepath) {
        try {
            File file = new File(filepath);
            if (!file.exists()) {
                Log.e(TAG, "File does not exist: " + filepath);
                return null;
            }

            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                // Skip the header row
                if (lineNumber == 1) {
//                    Log.e(TAG, "Skipping header row");
                    continue;
                }

                // Process the second row (first data frame)
                if (lineNumber == 2) {
//                    Log.e(TAG, "Processing first data frame");
                    float[][] frame = parseRow(line);
                    reader.close();
                    return frame;
                }
            }

            reader.close();
            Log.e(TAG, "No data rows found in CSV");
            return null;

        } catch (IOException e) {
            Log.e(TAG, "Error reading CSV file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Parse a single row from the CSV
     * @param row The CSV row as a string
     * @return 2D array [32][52] after reshaping and rotation
     */
    private static float[][] parseRow(String row) {
        // Split by comma
        String[] tokens = row.split(",");

        // Skip first two columns and collect the rest
        List<Float> values = new ArrayList<>();
        for (int i = 2; i < tokens.length; i++) {
            String token = tokens[i].trim();
            if (!token.isEmpty()) {
                try {
                    float value = Float.parseFloat(token);
                    values.add(value);
                } catch (NumberFormatException e) {
                    // Skip non-numeric values
                    Log.w(TAG, "Skipping non-numeric value: " + token);
                }
            }
        }

//        Log.e(TAG, "Parsed " + values.size() + " values from row");

        // Check if we have the expected number of elements (52 * 32 = 1664)
        int expectedSize = ROWS * COLS;
        if (values.size() != expectedSize) {
            Log.w(TAG, "Warning: Expected " + expectedSize + " values, but got " + values.size());
        }

        // Reshape vector to matrix in column-major order
        // Then rotate 90 degrees clockwise
        float[][] matrix = vectorToMatrixColFirstRot90(values);

        return matrix;
    }

    /**
     * Reshape vector (column major) then rotate 90° clockwise.
     * Following the Python logic from measure.py
     */
    private static float[][] vectorToMatrixColFirstRot90(List<Float> vector) {
        // Create matrix in column-major order
        // Original shape is ROWS x COLS (52 x 32)
        float[][] tempMatrix = new float[ROWS][COLS];

        int index = 0;
        // Fill column by column (column-major order)
        for (int col = 0; col < COLS; col++) {
            for (int row = 0; row < ROWS; row++) {
                if (index < vector.size()) {
                    tempMatrix[row][col] = vector.get(index);
                    index++;
                }
            }
        }

//        Log.e(TAG, "Created matrix of size " + ROWS + " x " + COLS);

        // Rotate 90 degrees clockwise
        // After rotation: new dimensions are COLS x ROWS (32 x 52)
        float[][] rotatedMatrix = new float[COLS][ROWS];
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                // 90° clockwise rotation: rotated[j][ROWS-1-i] = original[i][j]
                rotatedMatrix[j][ROWS - 1 - i] = tempMatrix[i][j];
            }
        }

//        Log.e(TAG, "Rotated matrix to size " + COLS + " x " + ROWS);

        return rotatedMatrix;
    }

    /**
     * Helper to get minimum value from a 2D array
     */
    private static float getMin(float[][] matrix) {
        float min = Float.POSITIVE_INFINITY;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] < min) {
                    min = matrix[i][j];
                }
            }
        }
        return min;
    }

    /**
     * Helper to get mean value from a 2D array
     */
    private static float getMean(float[][] matrix) {
        float sum = 0;
        int count = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j];
                count++;
            }
        }
        return count > 0 ? sum / count : 0;
    }

    /**
     * Debug helper to print a portion of the matrix
     */
    public static void printMatrixDebug(float[][] matrix, int maxRows, int maxCols) {
        if (matrix == null) {
            Log.e(TAG, "Matrix is null");
            return;
        }

        int rows = Math.min(matrix.length, maxRows);
        int cols = (matrix.length > 0) ? Math.min(matrix[0].length, maxCols) : 0;

        Log.e(TAG, "Matrix dimensions: " + matrix.length + " x " +
                   ((matrix.length > 0) ? matrix[0].length : 0));
        Log.e(TAG, "Printing top-left " + rows + " x " + cols + " portion:");

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.setLength(0);
            sb.append("Row ").append(i).append(": ");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%8.2f ", matrix[i][j]));
            }
            Log.e(TAG, sb.toString());
        }
    }
}