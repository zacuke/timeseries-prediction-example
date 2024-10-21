using PandasNet;
using Tensorflow;
namespace timeseries_prediction_example;

public interface ITimeSeriesTask
{
    (IDatasetV2, IDatasetV2, IDatasetV2, Series, Series) GenerateDataset<TDataSource>(Func<TDataSource> ds);

    void Train(IDatasetV2 training_ds, IDatasetV2 val_ds, int epochs);

    float Test(IDatasetV2 dataset);

    Tensor Predict(Tensor input);

    void Config(string weightsPath, int inputWidth, int labelWidth, string[] labelColumns);
}

