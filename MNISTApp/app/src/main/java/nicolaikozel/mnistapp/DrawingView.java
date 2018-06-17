package nicolaikozel.mnistapp;

//DrawingView imports
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.util.Log;
import android.view.View;
import android.content.Context;
import android.util.AttributeSet;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.MotionEvent;

//DrawingView
public class DrawingView extends View {
    //Define paint - used to draw to canvas
    private Paint drawPaint, canvasPaint;
    //Define path - used to draw a path
    private Path drawPath;
    //Define bitmap - used to store the actual drawn image information
    private Bitmap bitmap;
    //Define canvas - used to render the bitmap
    private Canvas canvas;

    //Define matrix - used to scale bitmap up to canvas size
    private Matrix matrix;
    private Matrix invMatrix;

    //Define is setup flag - used to determine if drawing view is setup
    private boolean isSetup = false;

    //Constructor
    public DrawingView(Context context, AttributeSet attrs){
        super(context, attrs);
        Log.i("DEBUG","Creating DrawingView instance.");
        //Setup drawPaint to draw lines drawn by user
        drawPaint = new Paint();
        drawPaint.setColor(Color.BLACK);
        drawPaint.setAntiAlias(true);
        drawPaint.setStrokeWidth(1);
        drawPaint.setStyle(Paint.Style.STROKE);
        drawPaint.setStrokeJoin(Paint.Join.ROUND);
        drawPaint.setStrokeCap(Paint.Cap.ROUND);
        //Setup canvasPaint to draw bitmap to canvas
        canvasPaint = new Paint();
        //Initialize drawing path
        drawPath = new Path();
        //Initialize bitmap and canvas
        bitmap = Bitmap.createBitmap(MNISTDrawer.PIXEL_WIDTH, MNISTDrawer.PIXEL_HEIGHT, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        //Initialize matrices
        matrix = new Matrix();
        invMatrix = new Matrix();
    }

    //Clear canvas
    public void clear(){
        Log.i("DEBUG","Clearing canvas.");
        canvasPaint.setColor(Color.WHITE);
        canvas.drawRect(new Rect(0, 0, bitmap.getWidth(), bitmap.getHeight()), canvasPaint);
    }

    //Setup drawing view
    private void setup() {
        Log.i("DEBUG","Setting up DrawingView.");

        //View dimensions
        float width = getWidth();
        float height = getHeight();

        //Bitmap dimensions
        float bitmapWidth = MNISTDrawer.PIXEL_WIDTH;
        float bitmapHeight = MNISTDrawer.PIXEL_HEIGHT;

        //Calculate scale of bitmap compared to the view
        float scaleW = width / bitmapWidth;
        float scaleH = height / bitmapHeight;

        //Choose scale as the minimum of scaleW and scaleH
        float scale = scaleW;
        if (scale > scaleH) {
            scale = scaleH;
        }

        //Calculate canvas new x/y after scaling
        float newCanvasX = MNISTDrawer.PIXEL_WIDTH * scale / 2;
        float newCanvasY = MNISTDrawer.PIXEL_HEIGHT * scale / 2;
        float dx = width / 2 - newCanvasX;
        float dy = height / 2 - newCanvasY;

        //Apply scaling and translation to matrix
        matrix.setScale(scale, scale);
        matrix.postTranslate(dx, dy);
        matrix.invert(invMatrix);

        //Mark drawing view as setup
        isSetup = true;
    }

    public float[] getPixelData() {
        //Get dimensions of bitmap
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        //Get pixel information from bitmap
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        //Convert to 1 hot float array
        float[] retPixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            int b = pix & 0xff;
            retPixels[i] = (float) ((0xff - b) / 255.0);
        }
        return retPixels;
    }

    //Render paths and bitmap to canvas
    @Override
    protected void onDraw(Canvas canvas) {
        //Setup drawing
        if(!isSetup) {
            setup();
        }
        //Draw bitmap to canvas
        canvas.drawBitmap(bitmap, matrix, canvasPaint);
    }

    //Process user touching drawing view
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        //Get location user pressed on screen
        float touchX = event.getX();
        float touchY = event.getY();
        Log.i("DEBUG","onTouchEvent triggered: "+touchX+" "+touchY);

        //Convert press location to location relative to de-scaled bitmap
        float relX, relY;
        float[] tmpPoint = new float[]{touchX, touchY};
        invMatrix.mapPoints(tmpPoint);
        relX = tmpPoint[0];
        relY = tmpPoint[1];

        Log.i("DEBUG","Relative position: "+relX+" "+relY);

        //Process event type
        switch (event.getAction()) {
            //User pressed the screen
            case MotionEvent.ACTION_DOWN:
                drawPath.moveTo(relX, relY);
                break;
            //User touching and moving on screen
            case MotionEvent.ACTION_MOVE:
                drawPath.lineTo(relX, relY);
                canvas.drawPath(drawPath, drawPaint);
                break;
            //User stopped pressing the screen
            case MotionEvent.ACTION_UP:
                canvas.drawPath(drawPath, drawPaint);
                drawPath.reset();
                break;
            default:
                return false;
        }
        invalidate();
        return true;
    }
}