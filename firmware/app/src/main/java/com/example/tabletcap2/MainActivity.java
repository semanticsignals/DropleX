package com.example.tabletcap2;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

/**
 * DropleX — Raw Capacitance Heatmap
 *
 * Streams real-time delta values from the Samsung Galaxy Tab S6 Lite's
 * maXTouch capacitive sensor via the mxt-app command-line tool (run as root).
 * Each frame is rendered as a colour-coded 52 × 32 heatmap:
 *   green = negative delta, red = positive delta, white = near-zero / noise.
 *
 * Required device:  Samsung Galaxy Tab S6 Lite (SM-P610 / SM-P615)
 * Required binary:  /data/local/tmp/mxt-app  (built for arm64-v8a)
 * Required data:    /sdcard/logs/ref.csv      (reference baseline frame)
 * Root access:      yes — the app calls `su` to interact with the i²c device
 *
 * Controls:
 *   Volume Up     — toggle the red-cell overlay (electrode-position markers)
 *   Volume Down   — ignored (prevents accidental system-volume change)
 */
public class MainActivity extends AppCompatActivity implements TiltSensorManager.TiltListener {

    // ── Sensor device string for mxt-app ────────────────────────────────────
    private static final String DEV = "/data/local/tmp/mxt-app -d i2c-dev:0-004a --block-size 8";

    // How often the app polls a new sensor frame (milliseconds)
    private static final int CAPTURE_INTERVAL_MS = 2000;

    // ── UI ───────────────────────────────────────────────────────────────────
    private EditText       durationInput;
    private TextView       filenameDisplay;
    private TextView       tiltDisplay;

    // Background grid (drawn below the heatmap)
    private GridView       gridView;
    private android.widget.FrameLayout frameLayout;
    private LinearLayout   uiOverlay;

    // Heatmap overlay (added lazily on first frame)
    private HeatmapView    heatmapView;
    private float          originalBrightness = -1;

    // ── Tilt sensor ──────────────────────────────────────────────────────────
    private TiltSensorManager tiltSensorManager;

    // ── Session / capture loop ───────────────────────────────────────────────
    private String         sessionID;
    private java.io.File   sessionFolder;
    private Handler        captureHandler;
    private Runnable       captureRunnable;

    // ── Misc ─────────────────────────────────────────────────────────────────
    private Vibrator       vibrator;
    private boolean        vibStarted = false;
    private boolean        systemBrightnessSet = false;

    // ── Lifecycle ────────────────────────────────────────────────────────────

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!hasStoragePermission()) {
            requestStoragePermissions();
            setContentView(new TextView(this) {{
                setText("Storage permission required.\n\nPlease:\n1. Grant 'All files access' in Settings\n2. Restart the app");
                setTextSize(24);
                setGravity(android.view.Gravity.CENTER);
                setTextColor(Color.WHITE);
                setBackgroundColor(Color.BLACK);
                setPadding(50, 50, 50, 50);
            }});
            return;
        }

        // Full-screen, keep-awake
        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_FULLSCREEN | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        );
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        );
        if (getSupportActionBar() != null) getSupportActionBar().hide();

        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        setVolumeControlStream(android.media.AudioManager.STREAM_MUSIC);

        tiltSensorManager = new TiltSensorManager(this);
        tiltSensorManager.setTiltListener(this);

        buildLayout();
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (tiltSensorManager != null) {
            tiltSensorManager.startListening();
            new Handler().postDelayed(() -> {
                if (tiltSensorManager != null && tiltSensorManager.isListening()) {
                    tiltSensorManager.calibrate();
                }
            }, 500);
        }

        sessionID     = System.currentTimeMillis() + "";
        sessionFolder = createSessionFolder(sessionID);

        new Thread(() -> {
            try {
                Log.e("MainActivity", "Resetting and calibrating sensor…");
                executeResumeCommandsSync();

                if (sessionFolder != null) {
                    java.io.File delta0 = new java.io.File(sessionFolder, "deltas_0.csv");
                    Process p  = Runtime.getRuntime().exec("su");
                    java.io.DataOutputStream os = new java.io.DataOutputStream(p.getOutputStream());
                    os.writeBytes(DEV + " --debug-dump \"" + delta0.getAbsolutePath() + "\" --frames 1 --format 0\n");
                    os.writeBytes("exit\n");
                    os.flush();
                    os.close();
                    p.waitFor();
                }

                if (!systemBrightnessSet) {
                    setSystemBrightnessMax();
                    systemBrightnessSet = true;
                }

                runOnUiThread(this::startCapture);

            } catch (Exception e) {
                Log.e("MainActivity", "Init error: " + e.getMessage());
            }
        }).start();
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopCapture();
        executeStopCommands();
        if (tiltSensorManager != null) tiltSensorManager.stopListening();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopCapture();
        if (tiltSensorManager != null) {
            tiltSensorManager.stopListening();
            tiltSensorManager = null;
        }
    }

    // ── Capture loop ─────────────────────────────────────────────────────────

    private void startCapture() {
        if (captureRunnable != null) return;
        captureHandler   = new Handler();
        captureRunnable  = new Runnable() {
            @Override public void run() {
                executeCapture();
                captureHandler.postDelayed(this, CAPTURE_INTERVAL_MS);
            }
        };
        captureHandler.post(captureRunnable);
        Log.e("MainActivity", "Capture started (every " + CAPTURE_INTERVAL_MS + " ms)");
    }

    private void stopCapture() {
        if (captureRunnable != null && captureHandler != null) {
            captureHandler.removeCallbacks(captureRunnable);
            captureRunnable = null;
        }
    }

    /**
     * Issue one mxt-app delta-dump command as root, then hand the file off
     * to the CSV parser and heatmap renderer.
     */
    private void executeCapture() {
        new Thread(() -> {
            try {
                java.io.File targetDir = resolveTargetDir();
                String ts        = System.currentTimeMillis() + "";
                String deltaPath = new java.io.File(targetDir, "deltas_" + ts + ".csv").getAbsolutePath();
                String refPath   = new java.io.File(resolveLogsDir(), "ref.csv").getAbsolutePath();

                Process p  = Runtime.getRuntime().exec("su");
                java.io.DataOutputStream os = new java.io.DataOutputStream(p.getOutputStream());
                os.writeBytes(DEV + " --debug-dump \"" + deltaPath + "\" --frames 1 --format 0\n");
                os.writeBytes("exit\n");
                os.flush();
                os.close();

                if (p.waitFor() == 0) {
                    java.io.File recent = findMostRecentFile(targetDir, "deltas_");
                    if (recent != null && recent.exists()) processFrame(recent.getAbsolutePath(), refPath);
                }
            } catch (Exception e) {
                Log.e("MainActivity", "Capture error: " + e.getMessage());
            }
        }).start();
    }

    private void processFrame(String deltaPath, String refPath) {
        new Thread(() -> {
            try {
                float[][] frame = CSVFrameParser.parseFirstFrameNormalized(deltaPath, refPath);
                if (frame != null) {
                    displayHeatmap(transposeMatrix(frame));
                    runOnUiThread(() -> filenameDisplay.setText(
                            "Live  " + new java.text.SimpleDateFormat("HH:mm:ss").format(new java.util.Date())));
                }
            } catch (Exception e) {
                Log.e("MainActivity", "Frame parse error: " + e.getMessage());
            }
        }).start();
    }

    // ── Rendering ────────────────────────────────────────────────────────────

    private void displayHeatmap(final float[][] data) {
        runOnUiThread(() -> {
            // Set full brightness on first frame
            if (originalBrightness == -1) {
                WindowManager.LayoutParams lp = getWindow().getAttributes();
                originalBrightness = lp.screenBrightness;
                lp.screenBrightness = 1f;
                lp.flags |= WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON;
                getWindow().setAttributes(lp);
            }

            // Create HeatmapView lazily
            if (heatmapView == null) {
                heatmapView = new HeatmapView(MainActivity.this);
                android.widget.FrameLayout.LayoutParams p = new android.widget.FrameLayout.LayoutParams(
                        android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                        android.widget.FrameLayout.LayoutParams.MATCH_PARENT);
                heatmapView.setLayoutParams(p);
                heatmapView.setDebugMode(true);   // colour every cell
                heatmapView.setShowValues(true);  // print numeric values
                frameLayout.addView(heatmapView);
                if (uiOverlay != null) uiOverlay.bringToFront();
            }

            heatmapView.setData(data);
            heatmapView.post(heatmapView::clearAutoDetection);
        });
    }

    // ── Hardware shell commands ───────────────────────────────────────────────

    private void executeResumeCommandsSync() {
        try {
            Process p  = Runtime.getRuntime().exec("su");
            java.io.DataOutputStream os = new java.io.DataOutputStream(p.getOutputStream());
            os.writeBytes(DEV + " --reset\n");
            os.writeBytes(DEV + " -W -T18 -r1 02\n");
            os.writeBytes(DEV + " -W -T6 -r5 10\n");
            os.writeBytes(DEV + " --calibrate\n");
            os.writeBytes("exit\n");
            os.flush();
            os.close();
            p.waitFor();
        } catch (Exception e) {
            Log.e("MainActivity", "Resume command error: " + e.getMessage());
        }
    }

    private void executeStopCommands() {
        try {
            Process p  = Runtime.getRuntime().exec("su");
            java.io.DataOutputStream os = new java.io.DataOutputStream(p.getOutputStream());
            os.writeBytes(DEV + " -W -T18 -r1 00\n");
            os.writeBytes(DEV + " -W -T6 -r0 01\n");
            os.writeBytes("exit\n");
            os.flush();
            os.close();
            p.waitFor();
        } catch (Exception e) {
            Log.e("MainActivity", "Stop command error: " + e.getMessage());
        }
    }

    private void setSystemBrightnessMax() {
        try {
            Process p  = Runtime.getRuntime().exec("su");
            java.io.DataOutputStream os = new java.io.DataOutputStream(p.getOutputStream());
            os.writeBytes("settings put system screen_brightness 255\n");
            os.writeBytes("exit\n");
            os.flush();
            os.close();
            p.waitFor();
        } catch (Exception e) {
            Log.e("MainActivity", "Brightness command error: " + e.getMessage());
        }
    }

    // ── Permissions ───────────────────────────────────────────────────────────

    private boolean hasStoragePermission() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            return android.os.Environment.isExternalStorageManager();
        }
        return androidx.core.content.ContextCompat.checkSelfPermission(this,
                android.Manifest.permission.READ_EXTERNAL_STORAGE)
                == android.content.pm.PackageManager.PERMISSION_GRANTED;
    }

    private void requestStoragePermissions() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            try {
                android.content.Intent intent = new android.content.Intent(
                        android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(android.net.Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            } catch (Exception e) {
                android.content.Intent intent = new android.content.Intent(
                        android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                startActivity(intent);
            }
        } else {
            androidx.core.app.ActivityCompat.requestPermissions(this,
                    new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE,
                                 android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }

    // ── File helpers ──────────────────────────────────────────────────────────

    private java.io.File resolveLogsDir() {
        java.io.File primary = new java.io.File("/sdcard/logs");
        if (primary.exists() && primary.canRead()) return primary;
        java.io.File backup = new java.io.File(getExternalFilesDir(null), "logs");
        if (!backup.exists()) backup.mkdirs();
        return backup;
    }

    private java.io.File resolveTargetDir() {
        if (sessionFolder != null && sessionFolder.exists()) return sessionFolder;
        return resolveLogsDir();
    }

    private java.io.File createSessionFolder(String id) {
        try {
            java.io.File dir = new java.io.File(resolveLogsDir(), "session_" + id);
            if (!dir.exists()) dir.mkdirs();
            return dir;
        } catch (Exception e) {
            return null;
        }
    }

    private java.io.File findMostRecentFile(java.io.File directory, String prefix) {
        if (directory == null || !directory.isDirectory()) return null;
        java.io.File[] files = directory.listFiles(f ->
                f.isFile() && f.getName().startsWith(prefix) && f.getName().endsWith(".csv"));
        if (files == null || files.length == 0) return null;
        java.io.File best = null;
        for (java.io.File f : files) {
            if (best == null || f.lastModified() > best.lastModified()) best = f;
        }
        return best;
    }

    private float[][] transposeMatrix(float[][] m) {
        if (m == null || m.length == 0) return m;
        int rows = m.length, cols = m[0].length;
        float[][] t = new float[cols][rows];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) t[j][i] = m[i][j];
        return t;
    }

    // ── Layout ────────────────────────────────────────────────────────────────

    private void buildLayout() {
        frameLayout = new android.widget.FrameLayout(this);

        // Background grid
        gridView = new GridView(this);
        frameLayout.addView(gridView, new android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT));

        // UI control strip at top
        uiOverlay = new LinearLayout(this);
        uiOverlay.setOrientation(LinearLayout.VERTICAL);
        android.widget.FrameLayout.LayoutParams overlayParams = new android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.WRAP_CONTENT);
        overlayParams.gravity = android.view.Gravity.TOP;
        uiOverlay.setLayoutParams(overlayParams);

        LinearLayout controlPanel = new LinearLayout(this);
        controlPanel.setOrientation(LinearLayout.HORIZONTAL);
        controlPanel.setPadding(16, 16, 16, 16);
        controlPanel.setBackgroundColor(Color.parseColor("#CC2C2C2C"));

        // Recording duration field
        TextView durationLabel = new TextView(this);
        durationLabel.setText("Time: ");
        durationLabel.setTextColor(Color.WHITE);
        durationLabel.setPadding(8, 8, 8, 8);

        durationInput = new EditText(this);
        durationInput.setText("10");
        durationInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        durationInput.setTextColor(Color.WHITE);
        durationInput.setBackgroundColor(Color.parseColor("#404040"));
        durationInput.setPadding(12, 12, 12, 12);
        LinearLayout.LayoutParams inputParams = new LinearLayout.LayoutParams(200,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        inputParams.setMargins(8, 0, 16, 0);
        durationInput.setLayoutParams(inputParams);

        // Live timestamp
        filenameDisplay = new TextView(this);
        filenameDisplay.setText("Starting…");
        filenameDisplay.setTextColor(Color.YELLOW);
        filenameDisplay.setPadding(16, 8, 8, 8);
        filenameDisplay.setTextSize(22);

        // Tilt readout
        tiltDisplay = new TextView(this);
        tiltDisplay.setText("Tilt: --");
        tiltDisplay.setTextColor(Color.GREEN);
        tiltDisplay.setTextSize(18);
        tiltDisplay.setPadding(16, 8, 8, 8);

        controlPanel.addView(durationLabel);
        controlPanel.addView(durationInput);
        controlPanel.addView(filenameDisplay);
        controlPanel.addView(tiltDisplay);

        uiOverlay.addView(controlPanel);
        frameLayout.addView(uiOverlay);

        setContentView(frameLayout);
    }

    // ── TiltListener ──────────────────────────────────────────────────────────

    @Override
    public void onTiltChanged(final float tiltAngle, float rawX, float rawY, float rawZ) {
        runOnUiThread(() -> {
            if (tiltDisplay != null) {
                tiltDisplay.setText(String.format("Tilt: %.1f°", tiltAngle));
                int color = Math.abs(tiltAngle) < 5  ? Color.GREEN
                          : Math.abs(tiltAngle) < 30 ? Color.CYAN
                          : Math.abs(tiltAngle) < 60 ? Color.YELLOW
                          : Color.RED;
                tiltDisplay.setTextColor(color);
            }
        });
    }

    // ── Volume key handling ───────────────────────────────────────────────────

    @Override
    public boolean onKeyDown(int keyCode, android.view.KeyEvent event) {
        if (keyCode == android.view.KeyEvent.KEYCODE_VOLUME_UP) {
            if (heatmapView != null) {
                heatmapView.toggleRedCells2();
                Log.e("MainActivity", "Red-cell overlay: " + heatmapView.isShowRedCells2());
            }
            return true;
        }
        if (keyCode == android.view.KeyEvent.KEYCODE_VOLUME_DOWN) {
            return true; // consume — no action
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    public boolean onKeyUp(int keyCode, android.view.KeyEvent event) {
        if (keyCode == android.view.KeyEvent.KEYCODE_VOLUME_UP
                || keyCode == android.view.KeyEvent.KEYCODE_VOLUME_DOWN) return true;
        return super.onKeyUp(keyCode, event);
    }

    @Override
    public boolean onKeyLongPress(int keyCode, android.view.KeyEvent event) {
        if (keyCode == android.view.KeyEvent.KEYCODE_VOLUME_UP
                || keyCode == android.view.KeyEvent.KEYCODE_VOLUME_DOWN) return true;
        return super.onKeyLongPress(keyCode, event);
    }

    // ── Background GridView (drawn behind the heatmap) ────────────────────────

    private class GridView extends View {
        private final Paint backgroundPaint = new Paint();

        public GridView(Context context) {
            super(context);
            backgroundPaint.setColor(Color.BLACK);
            backgroundPaint.setStyle(Paint.Style.FILL);
        }

        public void setBackgroundBrightness(int brightness) {
            int b = Math.max(0, Math.min(255, brightness));
            backgroundPaint.setColor(Color.rgb(b, b, b));
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            canvas.drawRect(0, 0, getWidth(), getHeight(), backgroundPaint);
        }
    }
}
