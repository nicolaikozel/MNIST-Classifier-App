package nicolaikozel.mnistapp;

//MNISTDrawer Imports
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View.OnClickListener;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

//MNISTDrawer
public class MNISTDrawer extends AppCompatActivity implements OnClickListener {
    //Constants
    public static final int PIXEL_WIDTH = 28;
    public static final int PIXEL_HEIGHT = 28;

    //Activity components
    private Button clearBtn, detectBtn;
    private DrawingView drawView;
    private TextView softmaxLabelText, cnnLabelText;

    // Tensorflow
    private TensorFlowInferenceInterface inferenceInterface_mnist;
    private TensorFlowInferenceInterface inferenceInterface_mnist_cnn;
    private static final String MNIST_MODEL = "file:///android_asset/frozen_mnist_model.pb";
    private static final String MNIST_MODEL_CNN = "file:///android_asset/frozen_mnist_model_cnn.pb";
    private static final String[] INPUT_NODES = {"modelInput"};
    private static final String[] OUTPUT_NODES = {"modelOutput"};
    private static final int[] INPUT_DIM = {1, PIXEL_WIDTH*PIXEL_HEIGHT};

    //Constructor for activity
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        //Setup mnistdrawer activity
        super.onCreate(savedInstanceState);
        Log.i("DEBUG","Creating MNISTDrawer instance.");
        setContentView(R.layout.activity_mnistdrawer);

        //Initialize MNIST model
        inferenceInterface_mnist = new TensorFlowInferenceInterface(getAssets(), MNIST_MODEL);
        inferenceInterface_mnist_cnn = new TensorFlowInferenceInterface(getAssets(), MNIST_MODEL_CNN);

        //Get buttons from activity and add click listeners
        clearBtn = (Button)findViewById(R.id.clear_btn);
        clearBtn.setOnClickListener(this);
        detectBtn = (Button)findViewById(R.id.detect_btn);
        detectBtn.setOnClickListener(this);

        //Get drawing view from activity
        drawView = (DrawingView)findViewById(R.id.drawing_view);

        //Get label text view from activity
        softmaxLabelText = (TextView)findViewById (R.id.softmax_label_text);
        cnnLabelText = (TextView)findViewById (R.id.cnn_label_text);
    }

    //Process button clicks on activity
    @Override
    public void onClick(View view){
        //Clear button clicked
        if(view.getId()==R.id.clear_btn){
            Log.i("DEBUG","Clear button pressed.");
            clear();
        //Detect button clicked
        }else if(view.getId()==R.id.detect_btn){
            Log.i("DEBUG","Detect button pressed.");
            detect();
        }
    }

    //Clear drawing view
    private void clear(){
        Log.i("DEBUG","Clearing draw view.");
        drawView.clear();
        drawView.invalidate();
        softmaxLabelText.setText("Softmax Label:");
        cnnLabelText.setText("CNN Label:");
    }

    //Detect label for drawn image
    private void detect(){
        Log.i("DEBUG","Detecting drawn image.");
        //Get bitmap information as float array
        float pixels[] = drawView.getPixelData();
        //Possible labels for image
        String[] labels = new String[]{"Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"};
        //Probability distribution given by trained model
        float[] modelOutputMnist = new float[labels.length];
        float[] modelOutputMnistCnn = new float[labels.length];

        //Feed in bitmap information
        inferenceInterface_mnist.feed(INPUT_NODES[0],pixels,INPUT_DIM[0], INPUT_DIM[1]);
        inferenceInterface_mnist_cnn.feed(INPUT_NODES[0],pixels,INPUT_DIM[0], INPUT_DIM[1]);
        float[] keep_prob = new float[1];
        keep_prob[0] = 1.0f;
        inferenceInterface_mnist_cnn.feed("keepProb",keep_prob,1,1);

        //Run session on frozen graph
        inferenceInterface_mnist.run(OUTPUT_NODES);
        inferenceInterface_mnist_cnn.run(OUTPUT_NODES);

        //Fetch probability distribution
        inferenceInterface_mnist.fetch(OUTPUT_NODES[0], modelOutputMnist);
        inferenceInterface_mnist_cnn.fetch(OUTPUT_NODES[0], modelOutputMnistCnn);

        //Find max value in probability distribution
        int indexMnist = 0;
        int indexMnistCnn = 0;
        for (int i=0; i<10; i++){
            if (modelOutputMnist[i] > modelOutputMnist[indexMnist]){
                indexMnist=i;
            }
            if (modelOutputMnistCnn[i] > modelOutputMnist[indexMnistCnn]){
                indexMnistCnn=i;
            }
        }
        //Display prediction
        Log.i("DEBUG", "Softmax prediction: "+indexMnist);
        softmaxLabelText.setText("Softmax Label: "+labels[indexMnist]);
        Log.i("DEBUG", "CNN prediction: "+indexMnistCnn);
        cnnLabelText.setText("CNN Label: "+labels[indexMnistCnn]);
    }

}
