using PandasNet;
using Tensorflow;
using Tensorflow.Keras.Engine;
namespace timeseries_prediction_example;

// https://www.tensorflow.org/tutorials/structured_data/time_series
public class ModelBase
{
    protected WindowGenerator _window;
    protected Model model;

    protected string _weightsPath;
    protected int _inputWidth;
    protected int _labelWidth;
    protected string[] _labelColumns;

    public (IDatasetV2, IDatasetV2, IDatasetV2, Series, Series) GenerateDataset<TDataSource>(Func<TDataSource> preprocess)
    {
        var ds = preprocess();
        if (ds is DataFrame df)
        {
            _window = new WindowGenerator(input_width: _inputWidth, 
                label_width: _labelWidth, 
                shift: 1,
                columns: df.columns,
                label_columns: _labelColumns);


            return _window.GenerateDataset(df);
        }
        else
            throw new NotImplementedException("");
    }

    public void Config(string weightsPath, int inputWidth, int labelWidth, string[] labelColumns)
    {
        _weightsPath = weightsPath;
        _inputWidth = inputWidth;
        _labelWidth = labelWidth;
        _labelColumns = labelColumns;
    }

    protected virtual Model BuildModel()
    {
        throw new NotImplementedException("");
    }

    public void Train(IDatasetV2 training_ds, IDatasetV2 val_ds, int epochs)
    {
        model = BuildModel();
        model.fit(training_ds, epochs: epochs, validation_data: val_ds);
        SaveModel();
    }

    protected void SaveModel()
    {
        if ( !string.IsNullOrEmpty(_weightsPath))
        {
            var filePath = new FileInfo(_weightsPath);
            Directory.CreateDirectory(filePath.Directory.FullName);
            model.save_weights(_weightsPath);
        }
    }

    public float Test(IDatasetV2 dataset)
    {
        if (model == null)
            model = BuildModel();
        foreach (var input in dataset.take(1))
            model.Apply(input);
        if ( !string.IsNullOrEmpty(_weightsPath))
            model.load_weights(_weightsPath);
        var result = model.evaluate(dataset);
        return result.First(x => x.Key == "mean_absolute_error").Value;
    }

    public Tensor Predict(Tensor input)
    {
        if (model == null)
            model = BuildModel();
        var result = model.predict(input);
        return result[0];
    }
}

