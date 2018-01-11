package com.dongguk.dm;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.dongguk.dm.env.ImageUtils;
import com.dongguk.dm.env.Logger;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

/**
 * Created by phong on 17. 11. 14.
 */

public class MyActivity extends Activity {
    final static String TAG = "Main Activity";
    static {
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV not loaded");
        } else {
            Log.d(TAG, "OpenCV loaded");
        }
    }
    private static final Logger LOGGER = new Logger();

    private static final int TF_OD_API_INPUT_SIZE = 300;
    //private static final String TF_OD_API_MODEL_FILE ="file:///android_asset/frozen_inference_graph.pb";
    private static final String TF_OD_API_MODEL_FILE =
            "file:///android_asset/frozen_inference_graph.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/marker_list.txt";
    private Classifier detector;
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private Bitmap croppedBitmap = null;
    private long lastProcessingTimeMs;
    private List<Classifier.Recognition> mappedRecognitions = new LinkedList<>();
    private List<Rect> lstRect = new LinkedList<>();
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private Bitmap rgbFrameBitmap = null;
    private int previewWidth = 1280;
    private int previewHeight = 720;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private static final String PERMISSION_READ = Manifest.permission.READ_EXTERNAL_STORAGE;
    private static final String PERMISSION_WRITE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static final int PERMISSIONS_REQUEST = 1;
    private boolean modelLoaded = false;
    private int count = 0;
    private int numberOfImageList = 0;
    private int numberOfDetectedBoundingbox = 0;
    private String filename="";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_myactivity);
        final Button btnloadModel = (Button)findViewById(R.id.btnLoad);
        final Button btnDetect= (Button)findViewById(R.id.btnDetect);
        final EditText txtTextResult = (EditText)findViewById(R.id.txtResult);
        final TextView txtStatus = (TextView)findViewById(R.id.txtStatus);

        btnloadModel.requestFocus();
        btnDetect.setEnabled(false);
        File externalStorageDir = Environment.getExternalStorageDirectory();




        btnloadModel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                LOGGER.d("Load model");
                AsyncTask.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            detector = TensorFlowObjectDetectionAPIModel.create(
                                    getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
                        } catch (final IOException e) {
                            LOGGER.e("Exception initializing classifier!", e);
                            Toast toast =
                                    Toast.makeText(
                                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                            toast.show();
                            finish();
                        }
                    }
                });
                modelLoaded = true;
                LOGGER.d("Loaded");
                LOGGER.d("------------------------------------------");
                txtStatus.setText("Model loaded !");
                btnDetect.setEnabled(true);
                btnloadModel.setEnabled(false);
            }
        });

        btnDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!modelLoaded)
                {
                    txtStatus.setText("Model not loaded !");
                }
                else
                {
                    count = 0;
                    numberOfImageList = 0;
                    btnDetect.setEnabled(false);
                    btnloadModel.setEnabled(false);

                    AsyncTask.execute(new Runnable() {
                        @Override
                        public void run() {
                            List<String> imageList = loadImage();
                            LOGGER.d("Found %d images", imageList.size());
                            LOGGER.d("------------------------------------------");
                            numberOfImageList = imageList.size();
                            frameToCropTransform =
                                    ImageUtils.getTransformationMatrix(
                                            TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                                            TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                                            0, true);
                            cropToFrameTransform = new Matrix();
                            frameToCropTransform.invert(cropToFrameTransform);

                            croppedBitmap = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);

                            for(String path: imageList) {
                                Uri imageURI = Uri.parse("file://" + path);
                                LOGGER.i(imageURI.getPath());
                                try {
                                    count ++;
                                    rgbFrameBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageURI);
                                    //resize image to 300x300 in order to match input size of CNN
                                    Bitmap resized = Bitmap.createScaledBitmap(rgbFrameBitmap,TF_OD_API_INPUT_SIZE,TF_OD_API_INPUT_SIZE,true);
                                    // crop image
                                    // actualy do nothing here
                                    final Canvas canvas = new Canvas(croppedBitmap);
                                    canvas.drawBitmap(resized, frameToCropTransform, null);

                                    // For examining the actual TF input.
                                    if (SAVE_PREVIEW_BITMAP) {
                                        ImageUtils.saveBitmap(croppedBitmap);
                                    }

                                    final long startTime = SystemClock.uptimeMillis();
                                    final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                                    LOGGER.i("Detect: %s", results);
                                    LOGGER.i("Processing time: %d",lastProcessingTimeMs);
                                    Bitmap copyBitmap = Bitmap.createBitmap(rgbFrameBitmap);
                                    Bitmap mutableCopyBitmap = copyBitmap.copy(Bitmap.Config.ARGB_8888, true);
                                    final Canvas copyCanvas = new Canvas(mutableCopyBitmap);
                                    final Paint paint = new Paint();
                                    paint.setColor(Color.RED);
                                    paint.setStyle(Paint.Style.STROKE);
                                    paint.setStrokeWidth(2.0f);


                                    float max_conf = 0;
                                    for (final Classifier.Recognition result : results) {
                                        final RectF location = result.getLocation();
                                        if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                                            numberOfDetectedBoundingbox++;
                                            RectF convertedRec = new RectF();
                                            convertedRec.left = location.left * previewWidth / TF_OD_API_INPUT_SIZE;
                                            convertedRec.top = location.top * previewHeight / TF_OD_API_INPUT_SIZE;
                                            convertedRec.right = location.right * previewWidth / TF_OD_API_INPUT_SIZE;
                                            convertedRec.bottom = location.bottom * previewHeight / TF_OD_API_INPUT_SIZE;
                                            copyCanvas.drawRect(convertedRec, paint);
                                            cropToFrameTransform.mapRect(convertedRec);
                                            result.setLocation(convertedRec);
                                        }
                                    }

                                    List<String> components = imageURI.getPathSegments();

                                    //Save result images
                                    filename = components.get(components.size() - 1).replace(".jpg",".png");
                                    saveBitmapResult(mutableCopyBitmap,filename,"DetectedResult_test");

                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            String result = String.format("Image(%s) Detected %d bounding box \n",filename,numberOfDetectedBoundingbox);
                                            txtTextResult.append(result);
                                            numberOfDetectedBoundingbox = 0;
                                            if(count<numberOfImageList)
                                                txtStatus.setText("Making detection !");
                                            else
                                                txtStatus.setText("Detection done !");
                                        }
                                    });
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }
                            LOGGER.d("Done.");
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast toast =
                                            Toast.makeText(
                                                    getApplicationContext(), "Detection job done !", Toast.LENGTH_LONG);
                                    toast.show();
                                }
                            });
                        }
                    });
                }
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();

        grantPermission();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onStop() {
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    /// Load model
    private void loadModel() {
        try {
            detector = TensorFlowObjectDetectionAPIModel.create(
                    getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        } catch (final IOException e) {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    /// Detect
    public static void saveBitmapResult(final Bitmap bitmap, final String filename, final String folderName) {
        final String root =
                Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + folderName;
        LOGGER.i("Saving %s %dx%d bitmap to %s.",filename, bitmap.getWidth(), bitmap.getHeight(), root);
        final File myDir = new File(root);

        if (!myDir.mkdirs()) {
            LOGGER.i("Make dir failed");
        }

        final String fname = filename;
        final File file = new File(myDir, fname);
        if (file.exists()) {
            file.delete();
        }
        try {
            final FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
            out.flush();
            out.close();
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
        }
    }

    private void detect(List<String> imageList) {

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        1280, 720,
                        0, true);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);


        croppedBitmap = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        for(String path: imageList) {
            Uri imageURI = Uri.parse("file://" + path);
            LOGGER.i(imageURI.getPath());
            try {
                processImageRGBbytes(imageURI);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /// Detect boundingbox in image
    protected void processImageRGBbytes(final Uri imageURI) throws IOException {
        rgbFrameBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageURI);
        //resize image to 300x300 in order to match input size of CNN
        Bitmap resized = Bitmap.createScaledBitmap(rgbFrameBitmap,300,300,true);
        // crop image
        // actualy do nothing here
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(resized, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        final long startTime = SystemClock.uptimeMillis();
        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        LOGGER.i("Detect: %s", results);
        LOGGER.i("Processing time: %d",lastProcessingTimeMs);
        Bitmap copyBitmap = Bitmap.createBitmap(rgbFrameBitmap);
        Bitmap mutableCopyBitmap = copyBitmap.copy(Bitmap.Config.ARGB_8888, true);
        final Canvas copyCanvas = new Canvas(mutableCopyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                RectF convertedRec = new RectF();
                convertedRec.left = location.left * 1280/300;
                convertedRec.top = location.top *720/300;
                convertedRec.right = location.right *1280/300;
                convertedRec.bottom = location.bottom *720/300;
                copyCanvas.drawRect(convertedRec, paint);
                cropToFrameTransform.mapRect(convertedRec);
                result.setLocation(convertedRec);

            }
        }
        List<String> components = imageURI.getPathSegments();
        String filename = components.get(components.size() - 1).replace(".jpg",".png");
        saveBitmapResult(mutableCopyBitmap,filename,"DetectedResult_test");

    }

    private List<String> loadImage() {

        if (!hasPermission()) {
            return null;
        }

        List<String> imageList = new ArrayList<>();

        Uri uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

        String[] projection = {MediaStore.MediaColumns.DATA, MediaStore.Images.Media.BUCKET_DISPLAY_NAME};

        final String orderBy = MediaStore.Images.Media.DATE_TAKEN;
        Cursor cursor = getApplicationContext().getContentResolver().query(uri, projection,
                null, null, orderBy + " ASC");

        int column_index_data = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA);
        int column_index_folder_name = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME);

        while (cursor.moveToNext()) {
            String folderName = cursor.getString(column_index_folder_name);
            if (!folderName.equals("Marker")) {
                continue;
            }
            String filePath = cursor.getString(column_index_data);
            imageList.add(filePath);
        }

        return imageList;
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_READ) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_WRITE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void grantPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_READ) ||
                    shouldShowRequestPermissionRationale(PERMISSION_WRITE)) {
                Toast.makeText(MyActivity.this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[]{PERMISSION_READ, PERMISSION_WRITE}, PERMISSIONS_REQUEST);
        }
    }



    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED
                        && grantResults[1] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    grantPermission();
                }
            }
        }
    }
}
