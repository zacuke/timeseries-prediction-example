using PandasNet;
using Tensorflow;
using static PandasNet.PandasApi;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using timeseries_prediction_example;

ITimeSeriesTask task;
IDatasetV2 training_ds, val_ds, test_ds;
Series train_mean, train_std;

task = new RnnModel();

task.Config(
    weightsPath: null, // @"timeseries_linear_v1\saved_weights.h5",
    inputWidth: 3,
    labelWidth: 1,
    labelColumns: new[] { "T (degC)" });

(training_ds, val_ds, test_ds, train_mean, train_std) = task.GenerateDataset(PrepareData);

//train
task.Train(training_ds, val_ds, epochs:1);

//test
Console.WriteLine("Starting Test");
var testResult = task.Test(test_ds);
Console.WriteLine("Test Result");
Console.WriteLine(testResult.ToString());

//predict
Console.WriteLine("Starting Prediction");
foreach (var (input, label) in test_ds.take(1))
{
    Tensor predictResult = task.Predict(input);
    Console.WriteLine("Predict Result");
    var denormalized = predictResult * train_std + train_mean;
    Console.WriteLine(denormalized.ToString());
}


DataFrame PrepareData()
{
    var zip_path = keras.utils.get_file("jena_climate_2009_2016.csv.zip",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        cache_subdir: "jena_climate_2009_2016",
        extract: true);

    var df = pd.read_csv(Path.Combine(zip_path, "jena_climate_2009_2016.csv"));
    // deal with hourly predictions, so start by sub-sampling the data from 10-minute intervals to one-hour intervals:
    df = df[new Slice(5, step: 6)];
    var date_time_string = df.pop("Date Time");
    var date_time = pd.to_datetime(date_time_string, "dd.MM.yyyy HH:mm:ss");
    print(df.head());

    // plot featuers
    /*var plot_cols = new string[] { "T (degC)", "p (mbar)", "rho (g/m**3)" };
    var plot_features = df[plot_cols];
    plot_features.index = date_time_string;
    plot_features.plot();*/

    print(df.describe().transpose());

    // Wind velocity
    var wv = df["wv (m/s)"];
    var bad_wv = wv == -9999.0f;
    wv[bad_wv] = 0.0f;

    var max_wv = df["max. wv (m/s)"];
    var bad_max_wv = max_wv == -9999.0f;
    max_wv[bad_max_wv] = 0.0f;

    // The above inplace edits are reflected in the DataFrame
    print(df["wv (m/s)"].min());

    // convert the wind direction and velocity columns to a wind vector
    wv = df.pop("wv (m/s)");
    max_wv = df.pop("max. wv (m/s)");

    // Convert to radians.
    var wd_rad = df.pop("wd (deg)") * pd.pi / 180;

    // Calculate the wind x and y components.
    df["Wx"] = wv * pd.cos(wd_rad);
    df["Wy"] = wv * pd.sin(wd_rad);

    // Calculate the max wind x and y components.
    df["max Wx"] = max_wv * pd.cos(wd_rad);
    df["max Wy"] = max_wv * pd.sin(wd_rad);

    var timestamp_s = date_time.map<DateTime, float>(pd.timestamp);

    var day = 24 * 60 * 60;
    var year = 365.2425f * day;
    df["Day sin"] = pd.sin(timestamp_s * (2 * pd.pi / day));
    df["Day cos"] = pd.cos(timestamp_s * (2 * pd.pi / day));
    df["Year sin"] = pd.sin(timestamp_s * (2 * pd.pi / year));
    df["Year cos"] = pd.cos(timestamp_s * (2 * pd.pi / year));

    return df;
}
 