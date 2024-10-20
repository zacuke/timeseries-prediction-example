using PandasNet;
using Tensorflow;


namespace timeseries_prediction_example;

public interface ITimeSeriesTask
{
    (IDatasetV2, IDatasetV2, IDatasetV2, Series, Series) GenerateDataset<TDataSource>(Func<TDataSource> ds);

    void Train(TrainingOptions options);

    void SetModelArgs<T>(T args);

    float Test(TestingOptions options);

    Tensor Predict(Tensor input);

    void Config(TaskOptions options);
}

