using Tensorflow;

namespace timeseries_prediction_example;

public class TrainingOptions
{

    public (IDatasetV2, IDatasetV2) Dataset { get; set; }
    public int Epochs { get; set; } = 5;
    //public int BatchSize { get; set; } = 100;
    //public int TrainingSteps { get; set; } = 100;
    //public float LearningRate { get; set; } = 0.001f;    

}
