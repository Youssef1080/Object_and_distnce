package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.hardware.Camera;
import android.speech.tts.TextToSpeech;
import android.text.TextUtils;
import android.util.Pair;
import android.util.Size;
import android.util.TypedValue;
import java.lang.Math;

import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Queue;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  private final HashMap<String,Float> dimentions = new HashMap<String,Float>();
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;
  float focalLength;
  float horizontalAngleView;
  private TextToSpeech tts;
  Camera  camera;
  private boolean sound = false;

  //add params
  Camera.Parameters params;

  public MultiBoxTracker(final Context context, int sizeWidth) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }
    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    dimentions.put("bird",0.10f);
    dimentions.put("cat" ,0.45f);
    dimentions.put("backpack" ,0.55f);
    dimentions.put("umbrella" ,0.50f);
    dimentions.put("bottle" ,0.20f);
    dimentions.put("wine glass" ,0.25f);
    dimentions.put("cup" ,0.15f);
    dimentions.put("fork" ,0.15f);
    dimentions.put("knife" ,0.25f);
    dimentions.put("spoon" ,0.15f);
    dimentions.put("banana" ,0.20f);
    dimentions.put("apple" ,0.07f);
    dimentions.put("sandwich" ,0.20f);
    dimentions.put("orange" ,0.08f);
    dimentions.put("chair" ,0.50f);
    dimentions.put("laptop" ,0.40f);
    dimentions.put("mouse" ,0.10f);
    dimentions.put("remote" ,0.20f);
    dimentions.put("keyboard" ,0.30f);
    dimentions.put("phone" ,0.15f);
    dimentions.put("book" ,0.18f);
    dimentions.put("toothbrush",0.16f);
    dimentions.put("person",1f);
    dimentions.put("car",0.8f);


    try {
      camera = Camera.open();
      camera.startPreview();
//      System.out.println("Detect: ");
      android.hardware.Camera.Parameters parameters;
      parameters = camera.getParameters();
//      focalLength = parameters.getFocalLength();
      //para
      float f = parameters.getFocalLength();
      horizontalAngleView = parameters.getHorizontalViewAngle();
      focalLength = (float) ((sizeWidth * 0.5) / Math.tan(horizontalAngleView * 0.5 * Math.PI/180));

      camera.stopPreview();
      camera.release();
      System.out.println("Focal: " + focalLength + " - sizeWidth: " + sizeWidth);

    } catch (RuntimeException ex) {
      // Here is your problem. Catching RuntimeException will make camera object null,
// so method 'getParameters();' won't work :)
      System.out.println("Fail: " + ex);
      android.hardware.Camera.Parameters parameters;
      parameters = camera.getParameters();
      focalLength = parameters.getFocalLength();
      System.out.println("Focal2: " + focalLength);
    }

    tts = new TextToSpeech(context, new TextToSpeech.OnInitListener() {
      @Override
      public void onInit(int status) {
        if(status != TextToSpeech.ERROR) {
          tts.setLanguage(Locale.UK);
        }
      }
    });

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }



    public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);

    // For Text to Speech
    float distanceClosest = 1000;
    String labelClosest = "";
    String positionCloset = "";

    // For each object
    float heightObject = 0;
    float distance = 0;
    String label;

    for (final TrackedRecognition recognition : trackedObjects) {

        final RectF trackedPos = new RectF(recognition.location);

        getFrameToCanvasMatrix().mapRect(trackedPos);
        boxPaint.setColor(recognition.color);

        float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
        canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

        int height = Resources.getSystem().getDisplayMetrics().heightPixels;
        int width = Resources.getSystem().getDisplayMetrics().widthPixels;


        String position = objectPosition(trackedPos, width);

        Float hashValue = dimentions.get(recognition.title);

        heightObject = hashValue == null ? 0.5f : hashValue;

        System.out.println("Object Inside: Height " + (trackedPos.bottom - trackedPos.top) + " - Width " + (trackedPos.right - trackedPos.left) + " - " + recognition.title);
        distance = distance_to_camera(heightObject, focalLength, trackedPos.bottom - trackedPos.top);

        label = recognition.title.substring(0, 1).toUpperCase() + recognition.title.substring(1);
        final String labelString =
                !TextUtils.isEmpty(recognition.title)
                        ? String.format("%s - %s", label, position, distance)
                        : String.format("%.2f", (100 * recognition.detectionConfidence));

        borderedText.drawText(
                canvas, trackedPos.left + cornerSize, trackedPos.top, labelString, boxPaint);
        borderedText.drawText(
                canvas, trackedPos.left + cornerSize, trackedPos.bottom, String.format("%.2f ft", distance), boxPaint);

        if (distance < distanceClosest) {
          distanceClosest = distance;
          labelClosest = label;
          positionCloset = position;
        }
    }
    // alert the closest object from the camera
    if (!sound && !"".equals(labelClosest)) {
      sound = true;
      makeSound("There is a " + labelClosest + " " + Math.round(distanceClosest) + " feet to your " + positionCloset);
    }

  }

  private void makeSound (String content) {
    new Thread(new Runnable() {
      @Override
      public void run() {
        try {

          tts.speak(content, TextToSpeech.QUEUE_ADD, null);
//          System.out.println("content: " + content);
          for (int i = 0; i < 15; i++) {
//            System.out.println("Shake: " + i);
            //Toast.makeText( getActivity(),"Cancel Alert in: " +i + " seconds", Toast.LENGTH_SHORT).show();
            Thread.sleep(200);

          }
          tts.stop();
          //tts.shutdown();
          sound = false;


        }
        catch (Exception e) {
          e.printStackTrace();
        }
      }
    }).start();
  }

  private float distance_to_camera(float knownWidth, float focalLength, float perWidth) {
      //compute and return the distance from the maker to the camera
      return (knownWidth * focalLength) / perWidth;
  }

  private String objectPosition (RectF trackedPos, int width) {
    String position = "";
    if ((trackedPos.right <= width / 2) ||
            (trackedPos.right - width / 2 < width / 8 && trackedPos.left < width / 4))
    {
      if (trackedPos.right <= width / 4)
        position = "Extreme Left";
      else
        position = "Left";
    }

    else if ((trackedPos.left >= width / 2) ||
            (width / 2 - trackedPos.left < width / 8 && width - trackedPos.right < width / 4)) {
      if (trackedPos.left >= (width - width / 4))
        position = "Extreme Right";
      else
        position = "Right";
    }
    else
      position = "Center";
    return position;
  }

  private void processResults(final List<Recognition> results) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedObjects.add(trackedRecognition);

      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }
}
