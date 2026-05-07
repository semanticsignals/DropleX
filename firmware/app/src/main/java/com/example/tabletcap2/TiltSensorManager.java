package com.example.tabletcap2;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

/**
 * Manages the tablet's tilt sensor using the accelerometer.
 * Calculates the tilt angle along the short axis of the tablet (left-right tilt).
 */
public class TiltSensorManager implements SensorEventListener {
    private static final String TAG = "TiltSensorManager";

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private TiltListener tiltListener;
    private boolean isListening = false;

    // Low-pass filter alpha (0 < alpha < 1)
    // Lower values = more smoothing but slower response
    private static final float FILTER_ALPHA = 0.8f;

    // Filtered accelerometer values
    private float[] gravity = new float[3];
    private boolean firstReading = true;

    // Calibration offset
    private float calibrationOffset = 0.0f;
    private boolean isCalibrated = false;

    /**
     * Interface for tilt updates
     */
    public interface TiltListener {
        void onTiltChanged(float tiltAngle, float rawX, float rawY, float rawZ);
    }

    public TiltSensorManager(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            if (accelerometer == null) {
                Log.e(TAG, "Accelerometer sensor not available on this device");
            }
        } else {
            Log.e(TAG, "SensorManager not available");
        }
    }

    /**
     * Set the listener for tilt updates
     */
    public void setTiltListener(TiltListener listener) {
        this.tiltListener = listener;
    }

    /**
     * Start listening to sensor updates
     */
    public void startListening() {
        if (accelerometer != null && !isListening) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
            isListening = true;
            firstReading = true;
            Log.d(TAG, "Started listening to accelerometer");
        }
    }

    /**
     * Stop listening to sensor updates
     */
    public void stopListening() {
        if (isListening) {
            sensorManager.unregisterListener(this);
            isListening = false;
            Log.d(TAG, "Stopped listening to accelerometer");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) {
            return;
        }

        // Apply low-pass filter to isolate gravity
        if (firstReading) {
            // Initialize with first reading
            gravity[0] = event.values[0];
            gravity[1] = event.values[1];
            gravity[2] = event.values[2];
            firstReading = false;
        } else {
            // Apply low-pass filter
            gravity[0] = FILTER_ALPHA * gravity[0] + (1 - FILTER_ALPHA) * event.values[0];
            gravity[1] = FILTER_ALPHA * gravity[1] + (1 - FILTER_ALPHA) * event.values[1];
            gravity[2] = FILTER_ALPHA * gravity[2] + (1 - FILTER_ALPHA) * event.values[2];
        }

        // Calculate tilt angle along the short axis (orthogonal to previous)
        // For a tablet in landscape mode:
        // X-axis: points to the right (short axis)
        // Y-axis: points up (long axis)
        // Z-axis: points out of the screen

        // Calculate tilt angle using X and Z components
        // This gives us the tilt along the short axis (rotation around Y-axis)
        float tiltRadians = (float) Math.atan2(gravity[0], gravity[2]);
        float tiltDegrees = (float) Math.toDegrees(tiltRadians);

        // Normalize to -90 to +90 degrees
        // 0° = flat on table (screen up)
        // +90° = tilted right edge up
        // -90° = tilted left edge up
        if (tiltDegrees > 90) {
            tiltDegrees = 180 - tiltDegrees;
        } else if (tiltDegrees < -90) {
            tiltDegrees = -180 - tiltDegrees;
        }

        // Apply calibration offset
        float calibratedTilt = tiltDegrees - calibrationOffset;

        // Keep the calibrated angle within -90 to +90 range
        if (calibratedTilt > 90) {
            calibratedTilt = calibratedTilt - 180;
        } else if (calibratedTilt < -90) {
            calibratedTilt = calibratedTilt + 180;
        }

        // Notify listener
        if (tiltListener != null) {
            tiltListener.onTiltChanged(calibratedTilt, gravity[0], gravity[1], gravity[2]);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not needed for this implementation
        Log.d(TAG, "Sensor accuracy changed: " + accuracy);
    }

    /**
     * Check if accelerometer is available
     */
    public boolean isAccelerometerAvailable() {
        return accelerometer != null;
    }

    /**
     * Check if currently listening
     */
    public boolean isListening() {
        return isListening;
    }

    /**
     * Calibrate the sensor to set current position as zero
     */
    public void calibrate() {
        // Calculate current tilt angle (using X and Z for short axis)
        float tiltRadians = (float) Math.atan2(gravity[0], gravity[2]);
        float tiltDegrees = (float) Math.toDegrees(tiltRadians);

        // Normalize to -90 to +90 degrees
        if (tiltDegrees > 90) {
            tiltDegrees = 180 - tiltDegrees;
        } else if (tiltDegrees < -90) {
            tiltDegrees = -180 - tiltDegrees;
        }

        // Store current angle as the calibration offset
        calibrationOffset = tiltDegrees;
        isCalibrated = true;

        Log.d(TAG, "Calibrated at angle: " + calibrationOffset + " degrees");
    }

    /**
     * Reset calibration to default (no offset)
     */
    public void resetCalibration() {
        calibrationOffset = 0.0f;
        isCalibrated = false;
        Log.d(TAG, "Calibration reset");
    }

    /**
     * Get current calibration offset
     */
    public float getCalibrationOffset() {
        return calibrationOffset;
    }

    /**
     * Check if sensor is calibrated
     */
    public boolean isCalibrated() {
        return isCalibrated;
    }
}